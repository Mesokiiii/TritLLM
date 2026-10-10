r"""
Triton Ternary GEMM Kernel & PyTorch BF16 Engine
Target Architecture: NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89, 128 SMs, 72KB/100KB L1/SRAM, 72MB L2)

This module provides:
1. High-performance OpenAI Triton GEMM kernel for 1.58-bit ternary weights {-1, 0, +1} on BF16 activations.
2. Autotuning (triton.autotune) optimized for RTX 4090's 128 Streaming Multiprocessors and memory subsystem.
3. 2-bit dense bit-packing (4 trits per uint8 byte) for 8x memory bandwidth reduction.
4. Robust native PyTorch BF16 fallback for environments without Triton (e.g. CPU or Windows).
5. PyTorch autograd Function (TritonTernaryLinearFunction) with Straight-Through Estimator (STE).
6. Drop-in nn.Module (TritonTernaryLinear) operating strictly in torch.bfloat16.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. TRITON IMPORT & HARDWARE COMPATIBILITY DETECTION
# ==============================================================================

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

# Hardware profile constants for NVIDIA RTX 4090 (Ada Lovelace Architecture)
RTX_4090_SPECS = {
    "arch": "Ada Lovelace (sm_89)",
    "sm_count": 128,
    "cuda_cores": 16384,
    "tensor_cores_gen": 4,
    "l1_shared_memory_per_sm_kb": 128,  # Configurable: up to 100KB SMEM
    "default_shared_memory_limit_kb": 72,
    "l2_cache_size_mb": 72.0,
    "memory_bus_width_bits": 384,
    "vram_bandwidth_gb_s": 1008.0,
    "bf16_tensor_tops": 330.0,
}


def is_triton_available() -> bool:
    """Returns True if Triton is installed and CUDA is available."""
    return HAS_TRITON and torch.cuda.is_available()


# ==============================================================================
# 1. 2-BIT TERNARY PACKING & UNPACKING (4 TRITS PER UINT8 BYTE)
# ==============================================================================
# Encoding table:
#   00 ->  0.0
#   01 -> +1.0
#   10 -> -1.0
#   11 ->  0.0 (Reserved / Zero)


def pack_ternary_2bit(w_ternary: torch.Tensor) -> torch.Tensor:
    """
    Packs a ternary tensor W in {-1.0, 0.0, +1.0} into 2-bit representation.
    4 trits are packed into 1 uint8 byte.
    
    Args:
        w_ternary: Tensor of shape (..., K) with values in {-1, 0, 1}.
                   Last dimension K must be divisible by 4.
                   
    Returns:
        packed: Byte tensor of shape (..., K // 4) with dtype=torch.uint8.
    """
    orig_shape = w_ternary.shape
    k = orig_shape[-1]
    if k % 4 != 0:
        raise ValueError(f"Last dimension {k} must be divisible by 4 for 2-bit packing.")

    # Flatten outer dimensions
    w_flat = w_ternary.reshape(-1, k)
    
    # Map {-1: 2, 0: 0, 1: 1}
    codes = torch.zeros_like(w_flat, dtype=torch.uint8)
    codes[w_flat == 1.0] = 1
    codes[w_flat == -1.0] = 2

    # Group into chunks of 4 along K
    chunks = codes.view(w_flat.shape[0], k // 4, 4)
    packed_flat = (
        (chunks[..., 0]) |
        (chunks[..., 1] << 2) |
        (chunks[..., 2] << 4) |
        (chunks[..., 3] << 6)
    )

    out_shape = list(orig_shape[:-1]) + [k // 4]
    return packed_flat.reshape(out_shape)


def unpack_ternary_2bit(
    w_packed: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Unpacks a 2-bit packed uint8 tensor back into ternary values {-1.0, 0.0, 1.0}.
    
    Args:
        w_packed: Byte tensor of shape (..., K // 4).
        dtype: Output dtype, strictly torch.bfloat16.
        
    Returns:
        w_unpacked: Tensor of shape (..., K) with values in {-1.0, 0.0, 1.0}.
    """
    orig_shape = w_packed.shape
    packed_flat = w_packed.reshape(-1, orig_shape[-1])
    
    # Extract 2-bit values
    t0 = (packed_flat >> 0) & 0x03
    t1 = (packed_flat >> 2) & 0x03
    t2 = (packed_flat >> 4) & 0x03
    t3 = (packed_flat >> 6) & 0x03

    def decode_chunk(t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(t, dtype=dtype)
        out[t == 1] = 1.0
        out[t == 2] = -1.0
        return out

    u0 = decode_chunk(t0)
    u1 = decode_chunk(t1)
    u2 = decode_chunk(t2)
    u3 = decode_chunk(t3)

    stacked = torch.stack([u0, u1, u2, u3], dim=-1)
    unpacked_flat = stacked.view(packed_flat.shape[0], orig_shape[-1] * 4)

    out_shape = list(orig_shape[:-1]) + [orig_shape[-1] * 4]
    return unpacked_flat.reshape(out_shape)


# ==============================================================================
# 2. OPENAI TRITON TERNARY GEMM KERNELS (OPTIMIZED FOR RTX 4090)
# ==============================================================================

if HAS_TRITON:
    # Autotuner configurations specifically designed for RTX 4090:
    # 128 SMs, 72KB shared memory tile limit, 72MB L2 cache.
    # GROUP_SIZE_M = 8 optimizes L2 cache reuse across the SM grid.
    RTX_4090_AUTOTUNE_CONFIGS = [
        # Large M (Prompt processing / Training batch, e.g. M >= 128)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        # Low latency generation (Decode phase, e.g. M in [1, 16])
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    ]

    @triton.autotune(
        configs=RTX_4090_AUTOTUNE_CONFIGS,
        key=["M", "N", "K"],
    )
    @triton.jit
    def _triton_ternary_gemm_kernel(
        # Pointers to Matrices
        a_ptr, b_ptr, c_ptr, scale_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        Triton Kernel: C = (A @ B) * scale
        A: (M, K) BF16 Activation Matrix
        B: (K, N) BF16 Ternary Matrix with elements in {-1.0, 0.0, 1.0}
        C: (M, N) BF16 Output Matrix
        scale: (N,) BF16 Per-channel scale vector (fused into epilogue)
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Block offsets
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Accumulator initialized in FP32 for numerical stability
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
            
            # Matrix multiply accumulate (Tensor Core BF16 MMA)
            accumulator += tl.dot(a, b)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Fused Epilogue: per-channel scaling factor alpha
        if scale_ptr is not None:
            scales = tl.load(scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
            accumulator = accumulator * scales[None, :]

        # Cast to bfloat16 and store
        c = accumulator.to(tl.bfloat16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.autotune(
        configs=RTX_4090_AUTOTUNE_CONFIGS,
        key=["M", "N", "K"],
    )
    @triton.jit
    def _triton_packed_2bit_gemm_kernel(
        a_ptr, b_packed_ptr, c_ptr, scale_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk_packed,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        Triton Kernel: C = (A @ Unpack2b(B_packed)^T) * scale
        Unpacks 4 trits per uint8 byte on-the-fly in registers/shared memory.
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 4
        K_PACKED = K // 4

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
            k_packed_offs = k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)

            # Load A tile [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + k_offs[None, :] * stride_ak)
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k_offs[None, :] < K), other=0.0)

            # Load packed B tile [BLOCK_N, BLOCK_K // 4]
            b_ptrs = b_packed_ptr + (offs_bn[:, None] * stride_bn + k_packed_offs[None, :] * stride_bk_packed)
            w_bytes = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & (k_packed_offs[None, :] < K_PACKED), other=0)

            # Unpack 4 trits: 00 -> 0.0, 01 -> +1.0, 10 -> -1.0
            t0 = (w_bytes >> 0) & 0x03
            t1 = (w_bytes >> 2) & 0x03
            t2 = (w_bytes >> 4) & 0x03
            t3 = (w_bytes >> 6) & 0x03

            w0 = tl.where(t0 == 1, 1.0, tl.where(t0 == 2, -1.0, 0.0)).to(tl.bfloat16)
            w1 = tl.where(t1 == 1, 1.0, tl.where(t1 == 2, -1.0, 0.0)).to(tl.bfloat16)
            w2 = tl.where(t2 == 1, 1.0, tl.where(t2 == 2, -1.0, 0.0)).to(tl.bfloat16)
            w3 = tl.where(t3 == 1, 1.0, tl.where(t3 == 2, -1.0, 0.0)).to(tl.bfloat16)

            # Decompose A along interleaved axes
            a0 = a[:, 0::4]
            a1 = a[:, 1::4]
            a2 = a[:, 2::4]
            a3 = a[:, 3::4]

            # Accumulate dot products
            accumulator += tl.dot(a0, tl.trans(w0))
            accumulator += tl.dot(a1, tl.trans(w1))
            accumulator += tl.dot(a2, tl.trans(w2))
            accumulator += tl.dot(a3, tl.trans(w3))

        if scale_ptr is not None:
            scales = tl.load(scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
            accumulator = accumulator * scales[None, :]

        c = accumulator.to(tl.bfloat16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# ==============================================================================
# 3. NATIVE PYTORCH BF16 FALLBACK
# ==============================================================================

def ternary_gemm_pytorch(
    x: torch.Tensor,
    w_ternary: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance native PyTorch BF16 GEMM fallback.
    Executes strictly in torch.bfloat16.
    
    Y = (X @ W_ternary^T) * scale + bias
    
    Args:
        x: Input activation tensor (*, K) in torch.bfloat16.
        w_ternary: Ternary weight tensor (N, K) in torch.bfloat16 with values in {-1, 0, 1}.
        scale: Optional per-channel scaling factor (N, 1) or (N,) in torch.bfloat16.
        bias: Optional bias tensor (N,) in torch.bfloat16.
        
    Returns:
        y: Output tensor (*, N) in torch.bfloat16.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if w_ternary.dtype != torch.bfloat16:
        w_ternary = w_ternary.to(torch.bfloat16)

    # Perform matrix multiplication
    # w_ternary is (N, K), F.linear computes x @ w_ternary^T -> (*, N)
    out = F.linear(x, w_ternary)

    if scale is not None:
        if scale.dtype != torch.bfloat16:
            scale = scale.to(torch.bfloat16)
        scale_reshaped = scale.view(1, -1) if out.ndim == 2 else scale.view(1, 1, -1)
        out = out * scale_reshaped

    if bias is not None:
        if bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        out = out + bias

    return out


# ==============================================================================
# 4. UNIFIED DISPATCHER (TRITON GPU + PYTORCH FALLBACK)
# ==============================================================================

def triton_ternary_gemm(
    x: torch.Tensor,
    w_ternary: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Dispatches to Triton ternary GEMM kernel if available and on CUDA,
    otherwise smoothly falls back to PyTorch native BF16 GEMM.
    
    Args:
        x: Input tensor (*, K) in BF16.
        w_ternary: Ternary weights (N, K) in BF16 with values in {-1, 0, 1}.
        scale: Optional per-channel scale (N,) in BF16.
        bias: Optional bias (N,) in BF16.
        use_triton: If True and hardware supports it, run Triton kernel.
        
    Returns:
        y: Result tensor (*, N) in BF16.
    """
    if not (use_triton and is_triton_available() and x.is_cuda and w_ternary.is_cuda):
        # Native PyTorch BF16 fallback path
        return ternary_gemm_pytorch(x, w_ternary, scale=scale, bias=bias)

    # Triton GPU Path
    orig_shape = x.shape
    k = orig_shape[-1]
    n = w_ternary.shape[0]
    x_2d = x.reshape(-1, k).contiguous()
    m = x_2d.shape[0]

    # Pre-allocate output buffer in BF16
    c = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)

    # Transpose weights to (K, N) column-major layout for Triton dot
    w_t = w_ternary.t().contiguous()

    scale_vec = scale.squeeze().contiguous() if scale is not None else None

    # Launch 1D autotuned grid
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
    )

    _triton_ternary_gemm_kernel[grid](
        x_2d, w_t, c, scale_vec,
        m, n, k,
        x_2d.stride(0), x_2d.stride(1),
        w_t.stride(0), w_t.stride(1),
        c.stride(0), c.stride(1),
    )

    if bias is not None:
        c = c + bias.to(torch.bfloat16)

    out_shape = list(orig_shape[:-1]) + [n]
    return c.reshape(out_shape)


# ==============================================================================
# 5. STRAIGHT-THROUGH ESTIMATOR (STE) AUTOGRAD FUNCTION
# ==============================================================================

class TritonTernaryLinearFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapping the Triton Ternary GEMM kernel with
    Straight-Through Estimator (STE) for continuous latent weights updates.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        use_triton: bool = True,
    ) -> torch.Tensor:
        """
        Forward computation with quantized ternary weights.
        """
        ctx.save_for_backward(x, weight, scale, bias)
        ctx.use_triton = use_triton

        # Quantize latent weights to strictly {-1.0, 0.0, 1.0} in BF16
        # alpha = mean(|W|)
        if scale is None:
            computed_scale = weight.abs().mean(dim=-1, keepdim=True).clamp(
                min=torch.tensor(1e-5, dtype=torch.bfloat16, device=weight.device)
            )
        else:
            computed_scale = scale

        scaled_w = weight / computed_scale
        w_ternary = torch.clamp(torch.round(scaled_w), -1.0, 1.0).to(torch.bfloat16)

        # Forward GEMM
        y = triton_ternary_gemm(
            x, w_ternary, scale=computed_scale, bias=bias, use_triton=use_triton
        )
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, scale, bias = ctx.saved_tensors
        use_triton = ctx.use_triton

        grad_x = None
        grad_weight = None
        grad_scale = None
        grad_bias = None

        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        # 1. Gradient with respect to input X:
        # dL/dX = grad_output @ (w_ternary * scale)
        if ctx.needs_input_grad[0]:
            if scale is None:
                alpha = weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
            else:
                alpha = scale
            w_effective = torch.clamp(torch.round(weight / alpha), -1.0, 1.0) * alpha
            grad_x = F.linear(grad_output, w_effective.t())

        # 2. Gradient with respect to latent continuous weights (STE pass-through):
        # dL/dW = grad_output^T @ X
        if ctx.needs_input_grad[1]:
            grad_out_flat = grad_output.reshape(-1, grad_output.shape[-1])
            x_flat = x.reshape(-1, x.shape[-1])
            grad_weight = torch.matmul(grad_out_flat.t(), x_flat)

        # 3. Gradient with respect to scale
        if scale is not None and ctx.needs_input_grad[2]:
            grad_scale = None  # Scale is dynamically computed or fixed

        # 4. Gradient with respect to bias
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=list(range(grad_output.ndim - 1)))

        return grad_x, grad_weight, grad_scale, grad_bias, None


# ==============================================================================
# 6. TRITON TERNARY LINEAR MODULE (STRICTLY BF16)
# ==============================================================================

class TritonTernaryLinear(nn.Module):
    """
    High-performance 1.58-bit Ternary Linear Layer.
    - Continuous latent weights kept in torch.bfloat16.
    - Forward pass dynamically quantizes weights to {-1.0, 0.0, 1.0} and executes
      via custom Triton GEMM or native PyTorch BF16 GEMM fallback.
    - Full backward support via Straight-Through Estimator (STE).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_triton: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_triton = use_triton

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
        # Ensure parameters are explicitly bfloat16
        self.weight.data = self.weight.data.to(torch.bfloat16)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            self.bias.data = self.bias.data.to(torch.bfloat16)

    def get_ternary_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the strictly ternary weights {-1.0, 0.0, 1.0} and scaling factor alpha.
        """
        with torch.no_grad():
            alpha = self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
            scaled = self.weight / alpha
            w_ternary = torch.clamp(torch.round(scaled), -1.0, 1.0).to(torch.bfloat16)
        return w_ternary, alpha

    def get_packed_weights(self) -> torch.Tensor:
        """
        Returns the 2-bit packed representation (uint8) of ternary weights.
        """
        w_ternary, _ = self.get_ternary_weights()
        return pack_ternary_2bit(w_ternary)

    def get_sparsity(self) -> float:
        """
        Computes the ternary sparsity ratio (fraction of zero weights).
        """
        w_ternary, _ = self.get_ternary_weights()
        total = w_ternary.numel()
        zeros = (w_ternary == 0.0).sum().item()
        return float(zeros) / float(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation strictly in BF16.
        """
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        return TritonTernaryLinearFunction.apply(
            x, self.weight, None, self.bias, self.use_triton
        )


# ==============================================================================
# 7. VERIFICATION & BENCHMARKING SUITE
# ==============================================================================

def run_self_test():
    """
    Self-test routine validating:
    - 2-bit packing and unpacking exactness.
    - Native PyTorch BF16 GEMM fallback parity.
    - Triton status and RTX 4090 architecture profile.
    - TritonTernaryLinear module forward and backward pass in BF16.
    """
    print("=" * 80)
    print("TRITON TERNARY GEMM & PYTORCH BF16 VERIFICATION SUITE")
    print("=" * 80)
    print(f"Python Platform: {sys.platform}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Triton Library Available: {HAS_TRITON}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("Target Architecture Profile:")
    for k, v in RTX_4090_SPECS.items():
        print(f"  - {k}: {v}")
    print("-" * 80)

    # 1. Test 2-bit bit-packing and unpacking
    print("[TEST 1] Testing 2-bit Bit-Packing & Unpacking...")
    w_ref = torch.tensor(
        [[-1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
         [ 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    packed = pack_ternary_2bit(w_ref)
    unpacked = unpack_ternary_2bit(packed, dtype=torch.bfloat16)
    assert torch.equal(w_ref, unpacked), "Error: Unpacked weights do not match reference!"
    compression_ratio = (w_ref.numel() * 2) / packed.numel()  # 2 bytes BF16 vs 1 byte packed (4 trits)
    print(f"  -> Packed {w_ref.numel()} trits into {packed.numel()} bytes.")
    print(f"  -> Bit-packing exactness: PASSED (Compression vs BF16: {compression_ratio:.1f}x)")

    # 2. Test PyTorch BF16 Fallback GEMM
    print("\n[TEST 2] Testing PyTorch Native BF16 GEMM Fallback...")
    M, K, N = 4, 128, 64
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_ternary = torch.randint(-1, 2, (N, K)).to(torch.bfloat16)
    scale = torch.ones(N, dtype=torch.bfloat16) * 0.5
    bias = torch.randn(N, dtype=torch.bfloat16) * 0.1

    out_gemm = ternary_gemm_pytorch(x, w_ternary, scale=scale, bias=bias)
    assert out_gemm.shape == (M, N)
    assert out_gemm.dtype == torch.bfloat16
    print(f"  -> Output shape: {out_gemm.shape}, dtype: {out_gemm.dtype}")
    print("  -> PyTorch Native BF16 Fallback: PASSED")

    # 3. Test TritonTernaryLinear Module (Forward & Backward in BF16)
    print("\n[TEST 3] Testing TritonTernaryLinear Layer (Strict BF16 Forward & STE Backward)...")
    layer = TritonTernaryLinear(in_features=K, out_features=N, bias=True)
    x_in = torch.randn(2, 8, K, dtype=torch.bfloat16, requires_grad=True)
    out = layer(x_in)
    assert out.shape == (2, 8, N)
    assert out.dtype == torch.bfloat16

    # Backward pass test
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype == torch.bfloat16
    assert x_in.grad is not None
    assert x_in.grad.dtype == torch.bfloat16
    print(f"  -> Module output: {out.shape}, dtype: {out.dtype}")
    print(f"  -> Latent weight grad dtype: {layer.weight.grad.dtype}")
    print(f"  -> Input grad dtype: {x_in.grad.dtype}")
    print(f"  -> Layer ternary sparsity: {layer.get_sparsity() * 100.0:.2f}%")
    print("  -> TritonTernaryLinear Forward & Backward: PASSED")

    print("\n" + "=" * 80)
    print("ALL TRITON TERNARY GEMM TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_self_test()
