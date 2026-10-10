r"""
TritFormer-850M: 1.58-bit Balanced Ternary Transformer Architecture
Reference Implementation and Mathematical Verification Suite

Architectural Highlights:
- Target parameter count: 800M - 900M (Target: ~850M).
  * Preset 850M Strict: d_model=2048, n_layers=16, n_heads=16, head_dim=128, d_ff=5632 -> 889,260,032 params (~889M).
  * Preset 24-Layer: d_model=2048, n_layers=24, n_heads=16, head_dim=128, d_ff=5632 -> 1,300,334,592 params (~1.30B).
- STRICT PROHIBITION OF FP32: All activations, normalizations (RMSNorm), embeddings, and RoPE tables operate strictly in torch.bfloat16.
- Weights are strictly ternary {-1.0, 0.0, +1.0} with per-channel scaling factors alpha (1.58-bit TernaryLinear with Straight-Through Estimator).
- High-efficiency Gated SwiGLU feed-forward network (d_ff = 5632).
- FlashAttention-2 support via PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention) with optional flash_attn backend.
- High-performance custom Triton GEMM engine integration (with PyTorch BF16 fallback).
- get_param_count() method providing complete parameter breakdown and ternary sparsity statistics.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt relative import from demsni package, fallback to absolute
try:
    from .triton_ternary_gemm import (
        TritonTernaryLinearFunction,
        is_triton_available,
        pack_ternary_2bit,
        ternary_gemm_pytorch,
        unpack_ternary_2bit,
    )
except ImportError:
    try:
        from triton_ternary_gemm import (
            TritonTernaryLinearFunction,
            is_triton_available,
            pack_ternary_2bit,
            ternary_gemm_pytorch,
            unpack_ternary_2bit,
        )
    except ImportError:
        # Fallback if triton_ternary_gemm is not found in path
        is_triton_available = lambda: False
        TritonTernaryLinearFunction = None
        pack_ternary_2bit = None
        unpack_ternary_2bit = None
        ternary_gemm_pytorch = None

# Import Fused Triton Kernels (Online LogSumExp, Fused RMSNorm, Fused SwiGLU, Static GEMM)
try:
    from .fused_kernels import (
        fused_cross_entropy,
        fused_swiglu,
        FusedRMSNormFunction,
        triton_ternary_gemm_static,
        ChunkedFusedLinearCrossEntropy,
    )
except ImportError:
    try:
        from fused_kernels import (
            fused_cross_entropy,
            fused_swiglu,
            FusedRMSNormFunction,
            triton_ternary_gemm_static,
            ChunkedFusedLinearCrossEntropy,
        )
    except ImportError:
        fused_cross_entropy = None
        fused_swiglu = None
        FusedRMSNormFunction = None
        triton_ternary_gemm_static = None
        ChunkedFusedLinearCrossEntropy = None

# Optional Flash Attention 2 library check
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN_LIB = True
except ImportError:
    HAS_FLASH_ATTN_LIB = False
    flash_attn_func = None


# ==============================================================================
# 1. ARCHITECTURE CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class TritFormer850MConfig:
    """
    Configuration specification for TritFormer-850M architecture.
    """
    # Vocabulary & Sequence Length
    vocab_size: int = 32768         # Target tokenizer vocabulary (or 50257)
    max_seq_len: int = 4096         # Context length window

    # Transformer Core Hyperparameters
    d_model: int = 2048             # Hidden dimension
    n_layers: int = 24              # Number of layers (24 as requested; 16 for strict 850M)
    n_heads: int = 16               # Number of attention heads
    head_dim: int = 128             # Dimension per head (16 * 128 = 2048 = d_model)
    d_ff: int = 5632                # Gated SwiGLU intermediate dimension

    # Positional Encoding & Normalization
    rope_theta: float = 10000.0     # Base frequency for RoPE
    norm_eps: float = 1e-5          # RMSNorm numerical stability constant in BF16

    # Regularization & Optimization
    dropout: float = 0.0            # Attention and residual dropout
    tie_word_embeddings: bool = True # Share embedding and LM head weights
    use_triton_gemm: bool = True    # Enable custom Triton GEMM kernel if available
    use_flash_attn: bool = True     # Enable FlashAttention-2 / SDPA

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * head_dim ({self.head_dim})"
        )

    @classmethod
    def for_850m(cls, vocab_size: int = 32768, **kwargs) -> TritFormer850MConfig:
        """
        Creates the canonical 850M parameter configuration (889,260,032 params).
        Using n_layers=16, d_model=2048, d_ff=5632, Gated SwiGLU.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=2048,
            n_layers=16,
            n_heads=16,
            head_dim=128,
            d_ff=5632,
            **kwargs,
        )

    @classmethod
    def for_24layers(cls, vocab_size: int = 32768, **kwargs) -> TritFormer850MConfig:
        """
        Creates the 24-layer configuration (1,300,334,592 params).
        Using n_layers=24, d_model=2048, d_ff=5632, Gated SwiGLU.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=2048,
            n_layers=24,
            n_heads=16,
            head_dim=128,
            d_ff=5632,
            **kwargs,
        )


# ==============================================================================
# 2. STRAIGHT-THROUGH ESTIMATOR (STE) QUANTIZER (STRICT BF16)
# ==============================================================================

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for 1.58-bit Ternary Quantization.
    Strictly preserves torch.bfloat16.
    
    Forward:
        alpha = mean(|W|, dim=-1, keepdim=True)
        W_scaled = W / alpha
        W_ternary = clamp(round(W_scaled), -1.0, 1.0)
        W_effective = W_ternary * alpha
        W_ste = W + (W_effective - W).detach()
        
    Backward:
        Gradients flow directly to latent continuous weights:
        dL/dW = dL/dW_ste
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        per_channel: bool = True,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        eps_tensor = torch.tensor(eps, dtype=torch.bfloat16, device=weight.device)

        if per_channel:
            alpha = weight.abs().mean(dim=-1, keepdim=True).clamp(min=eps_tensor)
        else:
            alpha = weight.abs().mean().clamp(min=eps_tensor)

        scaled = weight / alpha
        w_ternary = torch.clamp(torch.round(scaled), -1.0, 1.0).to(torch.bfloat16)

        # Eliminate potential IEEE -0.0
        w_ternary = torch.where(w_ternary == 0.0, torch.zeros_like(w_ternary), w_ternary)

        w_effective = w_ternary * alpha
        w_ste = weight + (w_effective - weight).detach()

        ctx.save_for_backward(weight, alpha)
        return w_ste, w_ternary, alpha

    @staticmethod
    def backward(ctx, grad_w_ste: torch.Tensor, grad_w_ternary: torch.Tensor, grad_alpha: torch.Tensor):
        # STE allows gradient to bypass discrete step directly into latent weights
        grad_weight = grad_w_ste.clone().to(torch.bfloat16)
        return grad_weight, None, None


# ==============================================================================
# 3. TERNARY LINEAR LAYER WITH TRITON / NATIVE BF16 DISPATCH
# ==============================================================================

class TernaryLinear(nn.Module):
    """
    1.58-bit Ternary Linear Projection Layer:
    - Continuous latent weights kept strictly in torch.bfloat16.
    - Forward pass dynamically quantizes weights to {-1.0, 0.0, 1.0} scaled by alpha.
    - Leverages custom Triton GEMM kernel when available on CUDA; otherwise dispatches
      to native PyTorch BF16 GEMM fallback.
    - Full autograd gradient support via Straight-Through Estimator.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        per_channel: bool = True,
        use_triton_gemm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_channel = per_channel
        self.use_triton_gemm = use_triton_gemm

        # Latent continuous weights strictly in BF16
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization adapted for ternary weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data.to(torch.bfloat16)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            self.bias.data = self.bias.data.to(torch.bfloat16)

    def get_ternary_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the strictly ternary weights {-1.0, 0.0, 1.0} and scaling factor alpha.
        """
        with torch.no_grad():
            _, w_ternary, alpha = TernaryQuantizeSTE.apply(
                self.weight, self.per_channel, 1e-5
            )
        return w_ternary, alpha

    def get_sparsity(self) -> float:
        """
        Returns the ternary sparsity ratio (fraction of 0.0 elements).
        """
        w_ternary, _ = self.get_ternary_weights()
        total = w_ternary.numel()
        zeros = (w_ternary == 0.0).sum().item()
        return float(zeros) / float(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation strictly in torch.bfloat16.
        """
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # If Triton is requested, available, and on CUDA
        if (
            self.use_triton_gemm
            and is_triton_available()
            and x.is_cuda
            and self.weight.is_cuda
        ):
            orig_shape = x.shape
            m = x.reshape(-1, orig_shape[-1]).shape[0]
            # Use static decode kernel for M <= 16 to eliminate autotune latency during autoregressive inference
            if m <= 16 and triton_ternary_gemm_static is not None and not self.training:
                w_ternary, alpha = self.get_ternary_weights()
                return triton_ternary_gemm_static(x, w_ternary, scale=alpha, bias=self.bias)
            elif TritonTernaryLinearFunction is not None:
                return TritonTernaryLinearFunction.apply(
                    x, self.weight, None, self.bias, True
                )

        # Standard accelerated PyTorch BF16 path with STE
        w_ste, _, _ = TernaryQuantizeSTE.apply(self.weight, self.per_channel, 1e-5)
        return F.linear(x, w_ste, self.bias)


# ==============================================================================
# 4. RMSNORM IN STRICT BF16 (NO FP32 CASTING)
# ==============================================================================

class RMSNormBF16(nn.Module):
    """
    Root Mean Square Normalization operating strictly in torch.bfloat16.
    STRICT PROHIBITION OF FP32: variance and scaling remain in BF16.
    
    Formula:
        y = (x / sqrt(mean(x^2) + eps)) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = torch.tensor(eps, dtype=torch.bfloat16)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        if (
            FusedRMSNormFunction is not None
            and is_triton_available()
            and x.is_cuda
            and self.weight.is_cuda
        ):
            return FusedRMSNormFunction.apply(x, self.weight, 1e-5)

        # Variance computed in BF16
        eps_device = self.eps.to(device=x.device, dtype=torch.bfloat16)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps_device)
        return (x_norm * self.weight).to(torch.bfloat16)


# ==============================================================================
# 5. ROTARY POSITION EMBEDDINGS (ROPE) IN STRICT BF16
# ==============================================================================

class RotaryEmbeddingBF16(nn.Module):
    """
    Rotary Position Embedding (RoPE) precomputed and applied strictly in torch.bfloat16.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency tables in BF16
        half = head_dim // 2
        freq_exponents = torch.arange(0, half, dtype=torch.bfloat16) / half
        freqs = 1.0 / (theta ** freq_exponents)

        t = torch.arange(max_seq_len, dtype=torch.bfloat16)
        angles = torch.outer(t, freqs)  # (max_seq_len, half)

        cos = torch.cos(angles).to(torch.bfloat16)
        sin = torch.sin(angles).to(torch.bfloat16)

        # Concatenate along last dimension to cover full head_dim: (1, 1, max_seq_len, head_dim)
        cos = torch.cat([cos, cos], dim=-1)[None, None, :, :]
        sin = torch.cat([sin, sin], dim=-1)[None, None, :, :]

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=torch.bfloat16)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=torch.bfloat16)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies 2D Rotary Embedding to query or key tensor of shape (B, H, T, D).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rx = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rx * sin).to(torch.bfloat16)


# ==============================================================================
# 6. TERNARY ATTENTION WITH FLASHATTENTION-2 / SDPA IN BF16
# ==============================================================================

class TritonTernaryAttention(nn.Module):
    """
    1.58-bit Ternary Multi-Head Attention (TMHA):
    - All Q, K, V and Out projections implemented as TernaryLinear layers in BF16.
    - Rotary Position Embeddings (RoPE) applied to Q and K in BF16.
    - Scaled Dot-Product Attention accelerated via PyTorch SDPA (native FlashAttention-2 / CuDNN)
      or flash_attn_func library if installed on CUDA.
    """

    def __init__(self, config: TritFormer850MConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = config.dropout
        self.use_flash_attn = config.use_flash_attn

        # Projections in strictly BF16
        self.q_proj = TernaryLinear(
            self.d_model, self.d_model, bias=False, use_triton_gemm=config.use_triton_gemm
        )
        self.k_proj = TernaryLinear(
            self.d_model, self.d_model, bias=False, use_triton_gemm=config.use_triton_gemm
        )
        self.v_proj = TernaryLinear(
            self.d_model, self.d_model, bias=False, use_triton_gemm=config.use_triton_gemm
        )
        self.out_proj = TernaryLinear(
            self.d_model, self.d_model, bias=False, use_triton_gemm=config.use_triton_gemm
        )

        # RoPE generator
        self.rotary_emb = RotaryEmbeddingBF16(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Linear ternary projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Rotary Position Embeddings (RoPE)
        cos, sin = self.rotary_emb(T, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3. Scaled Dot-Product Attention (FlashAttention-2 / SDPA)
        if (
            self.use_flash_attn
            and HAS_FLASH_ATTN_LIB
            and q.is_cuda
            and flash_attn_func is not None
        ):
            # Dedicated FlashAttention-2 kernel
            # Requires shape (B, T, H, D)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            attn_out = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=is_causal,
            )
            attn_out = attn_out.contiguous().view(B, T, C)
        else:
            # PyTorch SDPA (dispatches automatically to FlashAttention-2 on CUDA Ada/Ampere/Hopper)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask if not is_causal else None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal if mask is None else False,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # 4. Final Ternary Projection
        return self.out_proj(attn_out)


# ==============================================================================
# 7. GATED SWIGLU FEED-FORWARD NETWORK (d_ff = 5632)
# ==============================================================================

class TritonTernarySwiGLU(nn.Module):
    """
    Ternary Gated SwiGLU MLP:
    d_model = 2048, d_ff = 5632
    
    Formula:
        SwiGLU(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    
    All linear projections are strictly 1.58-bit TernaryLinear in torch.bfloat16.
    """

    def __init__(self, config: TritFormer850MConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.gate_proj = TernaryLinear(
            self.d_model, self.d_ff, bias=False, use_triton_gemm=config.use_triton_gemm
        )
        self.up_proj = TernaryLinear(
            self.d_model, self.d_ff, bias=False, use_triton_gemm=config.use_triton_gemm
        )
        self.down_proj = TernaryLinear(
            self.d_ff, self.d_model, bias=False, use_triton_gemm=config.use_triton_gemm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if fused_swiglu is not None and is_triton_available() and gate.is_cuda and up.is_cuda:
            hidden = fused_swiglu(gate, up)
        else:
            hidden = F.silu(gate) * up
        return self.down_proj(hidden)


# ==============================================================================
# 8. TRITFORMER TRANSFORMER BLOCK
# ==============================================================================

class TritFormerBlock850M(nn.Module):
    """
    TritFormer Transformer Block with Pre-LayerNorm (RMSNorm) and Residual Connections.
    
    Structure:
        x = x + Attention(RMSNorm1(x))
        x = x + SwiGLU(RMSNorm2(x))
    """

    def __init__(self, config: TritFormer850MConfig):
        super().__init__()
        self.norm1 = RMSNormBF16(config.d_model, eps=config.norm_eps)
        self.attn = TritonTernaryAttention(config)
        self.norm2 = RMSNormBF16(config.d_model, eps=config.norm_eps)
        self.mlp = TritonTernarySwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # Pre-LN Attention + Residual
        x = x + self.attn(self.norm1(x), mask=mask, is_causal=is_causal)
        # Pre-LN SwiGLU + Residual
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# 9. COMPLETE TRITFORMER-850M LANGUAGE MODEL
# ==============================================================================

class TritFormer850M(nn.Module):
    """
    Autoregressive Decoder-Only Language Model: TritFormer-850M.
    
    Specifications:
    - Target Parameters: ~850M (889M strict with 16 layers, 1.30B with 24 layers).
    - Activation & Norm precision: Strictly torch.bfloat16.
    - Weights: 1.58-bit ternary {-1, 0, +1} with STE and Triton GEMM support.
    - FlashAttention-2 / SDPA accelerated.
    - Gated SwiGLU (d_ff = 5632).
    """

    def __init__(self, config: Optional[TritFormer850MConfig] = None):
        super().__init__()
        if config is None:
            config = TritFormer850MConfig.for_850m()
        self.config = config

        # Token embedding table in BF16
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, dtype=torch.bfloat16
        )

        # Transformer blocks stack
        self.blocks = nn.ModuleList([
            TritFormerBlock850M(config) for _ in range(config.n_layers)
        ])

        # Final RMSNorm in BF16
        self.norm_f = RMSNormBF16(config.d_model, eps=config.norm_eps)

        # Language Model Head (logits)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = TernaryLinear(
                config.d_model, config.vocab_size, bias=False, use_triton_gemm=config.use_triton_gemm
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward computation pass.
        
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            targets: Optional LongTensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) in torch.bfloat16
            loss: Optional scalar cross-entropy loss
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Input sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Token embeddings strictly in BF16
        x = self.tok_embeddings(input_ids)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, is_causal=is_causal)

        # Final normalization in BF16
        x = self.norm_f(x)

        # Project to vocabulary logits
        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.tok_embeddings.weight)
        else:
            logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            if fused_cross_entropy is not None and is_triton_available() and logits.is_cuda and targets.is_cuda:
                loss = fused_cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.
        """
        self.eval()
        curr_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            idx_cond = curr_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]

            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits.float(), dim=-1).to(torch.bfloat16)
                next_token = torch.multinomial(probs.float(), num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return curr_ids

    def get_param_count(self, print_summary: bool = True) -> Dict[str, Union[int, float, Dict]]:
        """
        Calculates exact parameter counts, ternary sparsity ratio,
        and theoretical memory footprints.
        """
        total_params = 0
        ternary_params = 0
        dense_params = 0
        zero_ternary_count = 0
        ternary_layer_count = 0

        layer_breakdown = {}

        for name, module in self.named_modules():
            if isinstance(module, TernaryLinear):
                ternary_layer_count += 1
                w = module.weight
                numel = w.numel()
                ternary_params += numel
                total_params += numel

                # Calculate sparsity on quantized weights
                w_ternary, _ = module.get_ternary_weights()
                zeros = (w_ternary == 0.0).sum().item()
                zero_ternary_count += zeros
                layer_breakdown[name] = {
                    "params": numel,
                    "sparsity": float(zeros) / float(numel),
                    "type": "TernaryLinear (1.58-bit)",
                }
            elif isinstance(module, (nn.Embedding, RMSNormBF16)):
                for p_name, param in module.named_parameters(recurse=False):
                    numel = param.numel()
                    dense_params += numel
                    total_params += numel
                    layer_breakdown[f"{name}.{p_name}"] = {
                        "params": numel,
                        "type": "Dense Parameter (BF16)",
                    }

        # Calculate ternary sparsity ratio across the entire network
        ternary_sparsity = (
            float(zero_ternary_count) / float(ternary_params)
            if ternary_params > 0 else 0.0
        )

        # Theoretical memory calculations
        # Ternary weights: 1.585 bits/param (Shannon limit) or 2.0 bits/param (packed uint8)
        # Dense weights: 16 bits/param (BF16)
        memory_packed_1_58_mb = (
            (ternary_params * 1.585 + dense_params * 16.0) / (8.0 * 1024.0 * 1024.0)
        )
        memory_packed_2bit_mb = (
            (ternary_params * 2.0 + dense_params * 16.0) / (8.0 * 1024.0 * 1024.0)
        )
        memory_bf16_mb = (total_params * 2.0) / (1024.0 * 1024.0)
        memory_fp32_mb = (total_params * 4.0) / (1024.0 * 1024.0)

        compression_vs_bf16 = memory_bf16_mb / memory_packed_1_58_mb if memory_packed_1_58_mb > 0 else 1.0
        compression_vs_fp32 = memory_fp32_mb / memory_packed_1_58_mb if memory_packed_1_58_mb > 0 else 1.0

        stats = {
            "total_params": total_params,
            "total_params_millions": total_params / 1e6,
            "ternary_params": ternary_params,
            "dense_params": dense_params,
            "ternary_layer_count": ternary_layer_count,
            "ternary_sparsity_ratio": ternary_sparsity,
            "memory_packed_1_58bit_mb": memory_packed_1_58_mb,
            "memory_packed_2bit_mb": memory_packed_2bit_mb,
            "memory_bf16_mb": memory_bf16_mb,
            "memory_fp32_mb": memory_fp32_mb,
            "compression_ratio_vs_bf16": compression_vs_bf16,
            "compression_ratio_vs_fp32": compression_vs_fp32,
            "layer_breakdown": layer_breakdown,
        }

        if print_summary:
            print("=" * 80)
            print(f"TRITFORMER ARCHITECTURE PARAMETER & SPARSITY AUDIT")
            print("=" * 80)
            print(f"Model Configuration:")
            print(f"  - Hidden Dimension (d_model):     {self.config.d_model}")
            print(f"  - Layers Count (n_layers):        {self.config.n_layers}")
            print(f"  - Attention Heads (n_heads):      {self.config.n_heads} (head_dim = {self.config.head_dim})")
            print(f"  - SwiGLU Dimension (d_ff):        {self.config.d_ff}")
            print(f"  - Vocabulary Size:                {self.config.vocab_size}")
            print(f"  - Weight-Tied Embeddings:         {self.config.tie_word_embeddings}")
            print("-" * 80)
            print(f"Parameter Counts:")
            print(f"  - Total Parameters:               {total_params:,} ({total_params / 1e6:.2f}M)")
            print(f"  - 1.58-bit Ternary Parameters:    {ternary_params:,} ({ternary_params / total_params * 100:.2f}%)")
            print(f"  - Non-Ternary (Dense) Parameters: {dense_params:,} ({dense_params / total_params * 100:.2f}%)")
            print(f"  - Ternary Linear Projections:     {ternary_layer_count}")
            print(f"  - Overall Ternary Sparsity:       {ternary_sparsity * 100.0:.2f}% (zeros)")
            print("-" * 80)
            print(f"Memory Footprint & Theoretical Compression:")
            print(f"  - Baseline FP32 (Uncompressed):   {memory_fp32_mb:.2f} MB ({memory_fp32_mb / 1024:.2f} GB)")
            print(f"  - Native BF16 (Uncompressed):     {memory_bf16_mb:.2f} MB ({memory_bf16_mb / 1024:.2f} GB)")
            print(f"  - Packed 2-bit Storage:           {memory_packed_2bit_mb:.2f} MB")
            print(f"  - Theoretical 1.58-bit Storage:   {memory_packed_1_58_mb:.2f} MB")
            print(f"  - Compression Ratio vs FP32:      {compression_vs_fp32:.2f}x")
            print(f"  - Compression Ratio vs BF16:      {compression_vs_bf16:.2f}x")
            print("=" * 80)

        return stats


# ==============================================================================
# 10. LOCAL VERIFICATION & TEST RUNNER
# ==============================================================================

def run_local_verification():
    """
    Validates model instantiation, BF16 integrity, parameter calculations,
    and single-block forward/backward execution.
    """
    print("=" * 80)
    print("STARTING LOCAL VERIFICATION OF TRITFORMER-850M")
    print("=" * 80)

    # 1. Verify Configuration Parameter Math for Both 16-layer (~850M) and 24-layer Presets
    print("[STEP 1] Validating Parameter Math for Target Presets...")
    cfg_850m = TritFormer850MConfig.for_850m()
    cfg_24l = TritFormer850MConfig.for_24layers()

    def calculate_theoretical_params(cfg: TritFormer850MConfig) -> int:
        attn_layer = 4 * cfg.d_model * cfg.d_model
        mlp_layer = 3 * cfg.d_model * cfg.d_ff
        norm_layer = 2 * cfg.d_model
        block_params = attn_layer + mlp_layer + norm_layer
        emb_params = cfg.vocab_size * cfg.d_model
        head_params = 0 if cfg.tie_word_embeddings else cfg.vocab_size * cfg.d_model
        final_norm = cfg.d_model
        return cfg.n_layers * block_params + emb_params + head_params + final_norm

    params_850m = calculate_theoretical_params(cfg_850m)
    params_24l = calculate_theoretical_params(cfg_24l)

    print(f"  -> Preset 'for_850m' (n_layers=16): {params_850m:,} params ({params_850m / 1e6:.2f}M) [Target: 800-900M]")
    print(f"  -> Preset 'for_24layers' (n_layers=24): {params_24l:,} params ({params_24l / 1e6:.2f}M) [Target: 1.30B]")
    assert 800_000_000 <= params_850m <= 900_000_000, "Error: 850M preset outside 800-900M range!"
    print("  -> Configuration parameter math: PASSED")

    # 2. Test Single Block Instantiation and Forward/Backward in Strict BF16
    print("\n[STEP 2] Testing Single TritFormer Block (TritFormerBlock850M) in Strict BF16...")
    single_block = TritFormerBlock850M(cfg_850m)

    # Verify parameter dtypes
    for p_name, param in single_block.named_parameters():
        assert param.dtype == torch.bfloat16, f"FP32 detected in {p_name}! Dtype is {param.dtype}"
    print("  -> All parameters verified strictly in torch.bfloat16 (Zero FP32 leakage).")

    # Forward pass through single block
    batch_size = 2
    seq_len = 16
    x_test = torch.randn(batch_size, seq_len, cfg_850m.d_model, dtype=torch.bfloat16, requires_grad=True)
    out_block = single_block(x_test, is_causal=True)

    assert out_block.shape == (batch_size, seq_len, cfg_850m.d_model)
    assert out_block.dtype == torch.bfloat16
    print(f"  -> Forward pass output shape: {out_block.shape}, dtype: {out_block.dtype}")

    # Backward pass
    loss = out_block.sum()
    loss.backward()
    assert x_test.grad is not None and x_test.grad.dtype == torch.bfloat16
    print(f"  -> Backward pass grad dtype: {x_test.grad.dtype}")
    print("  -> Single block forward/backward: PASSED")

    # 3. Test Full Model Parameter Counting and Sparsity Method
    print("\n[STEP 3] Testing get_param_count() with 1-layer Test Model...")
    test_cfg = TritFormer850MConfig(
        vocab_size=1024,
        d_model=cfg_850m.d_model,
        n_layers=1,
        n_heads=cfg_850m.n_heads,
        head_dim=cfg_850m.head_dim,
        d_ff=cfg_850m.d_ff,
    )
    test_model = TritFormer850M(test_cfg)
    stats = test_model.get_param_count(print_summary=True)
    assert stats["total_params"] > 0
    assert 0.0 <= stats["ternary_sparsity_ratio"] <= 1.0
    print(f"  -> Measured Ternary Sparsity Ratio: {stats['ternary_sparsity_ratio'] * 100:.2f}%")
    print("  -> get_param_count() audit method: PASSED")

    # 4. Forward Pass with Discrete Token IDs through LM
    print("\n[STEP 4] Testing Autoregressive Forward Pass with Input IDs...")
    input_ids = torch.randint(0, test_cfg.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, test_cfg.vocab_size, (2, 8), dtype=torch.long)
    logits, loss = test_model(input_ids, targets=targets)
    assert logits.shape == (2, 8, test_cfg.vocab_size)
    assert logits.dtype == torch.bfloat16
    assert loss is not None
    print(f"  -> Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"  -> Loss: {loss.item():.4f}")
    print("  -> Autoregressive LM forward pass: PASSED")

    print("\n" + "=" * 80)
    print("ALL TRITFORMER-850M VERIFICATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_local_verification()
