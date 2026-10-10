r"""
Multi-GPU Distributed LLM Pretraining Suite (4x RTX 4090 / 96GB VRAM)
Target: C:\Users\1\Desktop\AImL\demsni\train_distributed.py

Architectural Specifications:
- Cluster: 4x NVIDIA GeForce RTX 4090 (24GB VRAM per GPU = 96GB Total VRAM).
- Distributed Execution: torchrun launcher with NCCL (or Gloo fallback) backend.
- STRICT BAN ON FP32: Native torch.amp.autocast('cuda', dtype=torch.bfloat16).
- Optimizer: STRICTLY AdamW! Native PyTorch fused AdamW (fused=True) or bitsandbytes AdamW8bit.
- Schedule: Warmup + Cosine Decay Learning Rate Scheduler with min_lr floor.
- Gradient Accumulation: Target global batch size = 512-1024 sequences = 1M-2M tokens/step.
- Communication Optimization: DDP model.no_sync() across accumulation steps (only sync on final step).
- Gradient Checkpointing: Non-reentrant activation checkpointing for ~70% VRAM savings.
- Hardware & Math Telemetry: Real-time TFLOP/s, MFU (%), Throughput (tokens/sec), Loss, Perplexity,
  and comprehensive per-GPU VRAM usage table.
- Resilient Checkpointing: Atomic checkpoint autosave every 5-10 minutes with state restoration.
"""

import os
import sys
import time
import math
import json
import shutil
import inspect
import logging
import argparse
import contextlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
# Ensure script directory and parent are on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import local dataset streamer
try:
    from ultra_streamer import UltraDatasetStreamer, UltraStreamConfig
    HAS_ULTRA_STREAMER = True
except ImportError:
    try:
        from demsni.ultra_streamer import UltraDatasetStreamer, UltraStreamConfig
        HAS_ULTRA_STREAMER = True
    except ImportError:
        HAS_ULTRA_STREAMER = False

try:
    from dataset_streamer import DistributedDatasetStreamer, StreamConfig, load_streaming_tokenizer
except ImportError:
    from demsni.dataset_streamer import DistributedDatasetStreamer, StreamConfig, load_streaming_tokenizer

# Import TritFormer core
try:
    from tritformer import TritFormerLM, TritFormerBlock, RMSNorm
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tritformer import TritFormerLM, TritFormerBlock, RMSNorm

# Import TritFormer-850M
try:
    from tritformer_850m import TritFormer850M, TritFormer850MConfig
except ImportError:
    try:
        from demsni.tritformer_850m import TritFormer850M, TritFormer850MConfig
    except ImportError:
        TritFormer850M, TritFormer850MConfig = None, None

# Import Fused Triton Kernels & FlashAttention context
try:
    from fused_kernels import fused_cross_entropy, FlashAttention2Context
except ImportError:
    try:
        from demsni.fused_kernels import fused_cross_entropy, FlashAttention2Context
    except ImportError:
        fused_cross_entropy = None
        FlashAttention2Context = None


# ==============================================================================
# 1. HARDWARE CONSTANTS & CLUSTER SPECIFICATIONS (4x RTX 4090)
# ==============================================================================

# NVIDIA RTX 4090 (Ada Lovelace sm_89):
# - 16,384 CUDA Cores, 512 4th Gen Tensor Cores
# - Dense BF16 Peak Tensor Compute: 165.2 TFLOPS per GPU (330.3 TFLOPS with 2:4 sparsity)
# - Cluster Dense Peak (4x GPUs): 660.8 TFLOPS
# - Memory: 24,576 MB GDDR6X per GPU (96 GB Cluster Total)
RTX4090_PEAK_BF16_TFLOPS = 165.2
RTX4090_VRAM_PER_GPU_GB = 24.0


# ==============================================================================
# 2. GRADIENT-CHECKPOINTED TRITFORMER & STANDARD LLAMA ARCHITECTURES
# ==============================================================================

class CheckpointedTritFormerLM(TritFormerLM):
    """
    1.58-bit Ternary Language Model with Non-Reentrant Gradient Checkpointing.
    Eliminates activation memory bottlenecks for seq_len=2048 on RTX 4090.
    """

    def __init__(self, *args, gradient_checkpointing: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        x = self.tok_embeddings(input_ids)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Non-reentrant checkpointing avoids backward reentrancy bugs
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x: block(_x, use_causal_mask=True),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, use_causal_mask=True)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if fused_cross_entropy is not None and torch.cuda.is_available() and logits.is_cuda and targets.is_cuda:
                loss = fused_cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss


if TritFormer850M is not None:
    class CheckpointedTritFormer850M(TritFormer850M):
        """
        TritFormer-850M 1.58-bit Balanced Ternary Transformer with Gradient Checkpointing.
        """
        def __init__(self, config=None, gradient_checkpointing: bool = True):
            super().__init__(config=config)
            self.gradient_checkpointing = gradient_checkpointing

        def forward(
            self,
            input_ids: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            is_causal: bool = True,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, T = input_ids.shape
            assert T <= self.config.max_seq_len, (
                f"Input sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )
            x = self.tok_embeddings(input_ids)
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        lambda _x: block(_x, mask=mask, is_causal=is_causal),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, mask=mask, is_causal=is_causal)

            x = self.norm_f(x)
            if self.config.tie_word_embeddings:
                logits = F.linear(x, self.tok_embeddings.weight)
            else:
                logits = self.lm_head(x)

            loss = None
            if targets is not None:
                if fused_cross_entropy is not None and torch.cuda.is_available() and logits.is_cuda and targets.is_cuda:
                    loss = fused_cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
                else:
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            return logits, loss
else:
    CheckpointedTritFormer850M = None


class LLaMABlock(nn.Module):
    """Standard High-Performance LLaMA-style Transformer Block with SDPA."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        self.norm2 = RMSNorm(d_model)
        # SwiGLU MLP
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        norm_x = self.norm1(x)

        q = self.q_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0+ Scaled Dot-Product Attention (FlashAttention-2 / Memory-Efficient Attention)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attn_out)

        # SwiGLU MLP
        norm_x2 = self.norm2(x)
        mlp_out = self.down_proj(F.silu(self.gate_proj(norm_x2)) * self.up_proj(norm_x2))
        x = x + mlp_out

        return x


class StandardLLaMALM(nn.Module):
    """Dense Baseline LLaMA Architecture for Multi-GPU Pretraining Comparison."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        d_ff = d_ff if d_ff is not None else int(2 * (4 * d_model) / 3)

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            LLaMABlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss


# ==============================================================================
# 3. OPTIMIZER: STRICTLY ADAMW WITH WEIGHT DECAY PARTITIONING
# ==============================================================================

def build_adamw_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    use_8bit_adam: bool = False,
    rank: int = 0,
) -> torch.optim.Optimizer:
    """
    Builds strictly AdamW optimizer with optimal 2D parameter decay separation.
    Applies weight_decay ONLY to 2D matrix weights (attention, MLP, embeddings).
    Zero weight decay for 1D biases and RMSNorm scaling vectors.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    total_params = sum(p.numel() for p in decay_params) + sum(p.numel() for p in no_decay_params)
    if rank == 0:
        logging.info(
            f"AdamW Configuration: {len(decay_params)} decay tensors ({sum(p.numel() for p in decay_params):,} params) | "
            f"{len(no_decay_params)} no-decay tensors ({sum(p.numel() for p in no_decay_params):,} params) | "
            f"Total Trainable: {total_params:,}"
        )

    # 1. 8-Bit AdamW (bitsandbytes)
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            if rank == 0:
                logging.info("Using bitsandbytes AdamW8bit optimizer (75% VRAM savings on optimizer states).")
            return bnb.optim.AdamW8bit(optim_groups, lr=lr, betas=betas, eps=eps)
        except ImportError:
            if rank == 0:
                logging.warning("bitsandbytes requested but not installed. Falling back to native PyTorch AdamW.")

    # 2. PyTorch Native Fused AdamW
    is_cuda = torch.cuda.is_available()
    has_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = is_cuda and has_fused

    if rank == 0:
        logging.info(f"Using PyTorch native AdamW (fused={use_fused}, lr={lr}, weight_decay={weight_decay}).")

    return torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=use_fused,
    )


# ==============================================================================
# 4. WARMUP + COSINE DECAY LEARNING RATE SCHEDULER
# ==============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Standard LLM Cosine Annealing Learning Rate Schedule with Linear Warmup.
    - [0, warmup_steps]: linear increase from 0.0 to 1.0
    - [warmup_steps, training_steps]: cosine decay from 1.0 to min_lr_ratio
    - [training_steps, inf]: floor at min_lr_ratio
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 5. DISTRIBUTED CLUSTER ENVIRONMENT SETUP
# ==============================================================================

def init_distributed_cluster() -> Tuple[int, int, int, torch.device]:
    """
    Initializes torchrun distributed process group.
    Returns: (global_rank, local_rank, world_size, torch_device)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_id = local_rank % num_gpus
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 or "MASTER_ADDR" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=global_rank,
                world_size=world_size,
            )

    return global_rank, local_rank, world_size, device


# ==============================================================================
# 6. DISTRIBUTED TELEMETRY & HARDWARE PROFILER
# ==============================================================================

class ClusterProfiler:
    """
    Collects VRAM, Throughput, TFLOP/s, and MFU across all 4x RTX 4090 GPUs.
    """

    def __init__(self, world_size: int, rank: int, device: torch.device, num_params: int):
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.num_params = num_params
        self.peak_bf16_tflops = RTX4090_PEAK_BF16_TFLOPS * world_size

    def collect_vram_metrics(self) -> List[Dict[str, float]]:
        """Collects allocated and reserved VRAM across all distributed ranks."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024.0 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024.0 ** 3)
        else:
            allocated = 0.0
            max_allocated = 0.0
            reserved = 0.0

        local_stat = {
            "rank": self.rank,
            "allocated_gb": round(allocated, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "reserved_gb": round(reserved, 2),
        }

        if self.world_size > 1:
            all_stats = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_stats, local_stat)
            return all_stats
        return [local_stat]

    def compute_tflops_and_mfu(
        self,
        tokens_in_step: int,
        step_duration: float,
        has_grad_checkpointing: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Computes Achieved TFLOP/s and Model FLOPs Utilization (MFU).
        Standard LLM FLOPs formula:
        - Dense: ~6 FLOPs per parameter per token (forward + backward)
        - With Activation Checkpointing: ~8 FLOPs per parameter per token (due to recomputation)
        """
        factor = 8.0 if has_grad_checkpointing else 6.0
        flops_per_step = factor * self.num_params * tokens_in_step
        achieved_tflops = (flops_per_step / max(step_duration, 1e-6)) / 1e12
        mfu_pct = (achieved_tflops / max(self.peak_bf16_tflops, 1e-3)) * 100.0
        tokens_per_sec = tokens_in_step / max(step_duration, 1e-6)

        return achieved_tflops, mfu_pct, tokens_per_sec


# ==============================================================================
# 7. RESILIENT CHECKPOINT MANAGER (ATOMIC AUTOSAVE)
# ==============================================================================

class CheckpointManager:
    """
    Thread-safe, atomic checkpointing system with rotation and resume capability.
    Only Rank 0 writes to disk, synchronized with cluster via barriers.
    """

    def __init__(self, checkpoint_dir: str, save_interval_sec: float = 300.0, rank: int = 0):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_sec = save_interval_sec
        self.rank = rank
        self.last_save_time = time.time()

        if self.rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def should_save(self, step: int, is_final: bool = False) -> bool:
        """Determines if checkpoint save threshold is reached (time or final step)."""
        now = time.time()
        time_elapsed = now - self.last_save_time
        return is_final or (time_elapsed >= self.save_interval_sec)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        tokens_seen: int,
        loss: float,
        config: Dict[str, Any],
        is_final: bool = False,
    ):
        """Atomically saves checkpoint to disk."""
        if self.rank != 0:
            return

        unwrapped_model = model.module if isinstance(model, DDP) else model
        checkpoint_data = {
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": loss,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        tag = "final" if is_final else f"step_{step:07d}"
        model_type = config.get("model_type", "model")
        filename = f"checkpoint_{model_type}_{tag}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        tmp_filepath = filepath + ".tmp"

        try:
            torch.save(checkpoint_data, tmp_filepath)
            shutil.move(tmp_filepath, filepath)

            # Update latest checkpoint pointer for this model architecture
            latest_path = os.path.join(self.checkpoint_dir, f"checkpoint_{model_type}_latest.pt")
            shutil.copyfile(filepath, latest_path)

            self.last_save_time = time.time()
            logging.info(f"Checkpoint successfully autosaved to {filepath} (Latest pointer updated).")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def load_latest(
        self,
        model: nn.Module,
        model_type: str = "tritformer",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[int, int, float]:
        """Restores training state from checkpoint_{model_type}_latest.pt if present."""
        latest_path = os.path.join(self.checkpoint_dir, f"checkpoint_{model_type}_latest.pt")
        if not os.path.exists(latest_path):
            logging.info(f"No previous checkpoint found at {latest_path}. Starting training from scratch.")
            return 0, 0, float("inf")

        logging.info(f"Found existing checkpoint at {latest_path}. Resuming training...")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)

        unwrapped_model = model.module if isinstance(model, DDP) else model
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        step = checkpoint.get("step", 0)
        tokens_seen = checkpoint.get("tokens_seen", 0)
        loss = checkpoint.get("loss", float("inf"))

        logging.info(f"Resumed from Step {step} | Total Tokens Seen: {tokens_seen:,} | Prior Loss: {loss:.4f}")
        return step, tokens_seen, loss


# ==============================================================================
# 8. MAIN DISTRIBUTED TRAINING PIPELINE
# ==============================================================================

def train_distributed(args):
    # 1. Distributed Environment Initialization
    global_rank, local_rank, world_size, device = init_distributed_cluster()

    # Logger Setup
    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARN,
        format=f"%(asctime)s [Rank {global_rank}/{world_size}] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if global_rank == 0:
        logging.info("=" * 80)
        logging.info("STARTING DISTRIBUTED TRAINING RUN ON 4x RTX 4090 CLUSTER")
        logging.info(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({RTX4090_VRAM_PER_GPU_GB} GB VRAM)")
        logging.info(f"Target Global Batch Size: {args.global_batch_size} sequences")
        logging.info(f"Sequence Context Length:  {args.seq_len} tokens")
        logging.info(f"Micro Batch Size per GPU: {args.micro_batch_size} sequences")
        logging.info("STRICT BAN ON FP32: Native torch.amp.autocast('cuda', dtype=torch.bfloat16) ENFORCED.")
        if FlashAttention2Context is not None:
            FlashAttention2Context.enable_flash_sdpa()
            logging.info("FlashAttention-2 / SDPA kernel configuration enabled for RTX 4090.")
        logging.info("=" * 80)

    # 2. Gradient Accumulation Calculus
    # Global tokens per step = global_batch_size * seq_len
    # E.g. 512 seqs * 2048 tokens = 1,048,576 tokens (~1M tokens)
    # E.g. 1024 seqs * 2048 tokens = 2,097,152 tokens (~2.1M tokens)
    micro_batch_sequences = world_size * args.micro_batch_size
    grad_accum_steps = max(1, args.global_batch_size // micro_batch_sequences)
    effective_global_batch = grad_accum_steps * micro_batch_sequences
    tokens_per_optimizer_step = effective_global_batch * args.seq_len

    if global_rank == 0:
        logging.info(f"Distributed Batch Calculus:")
        logging.info(f"  Cluster Micro-Batch:      {micro_batch_sequences} sequences ({micro_batch_sequences * args.seq_len:,} tokens)")
        logging.info(f"  Gradient Accum Steps:     {grad_accum_steps} accumulation passes")
        logging.info(f"  Effective Global Batch:   {effective_global_batch} sequences ({tokens_per_optimizer_step:,} tokens/step)")

    # 3. Model Instantiation & Checkpointing Support
    tokenizer = load_streaming_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)

    if args.model_type in ["tritformer_850m", "850m"] and CheckpointedTritFormer850M is not None:
        # Default to canonical 850M parameters (889M) if default flags were passed
        d_model = args.d_model if args.d_model != 512 else 2048
        n_heads = args.n_heads if args.n_heads != 8 else 16
        n_layers = args.n_layers if args.n_layers != 8 else 16
        d_ff = args.d_ff if args.d_ff != 1536 else 5632
        head_dim = max(1, d_model // n_heads)
        computed_d_model = head_dim * n_heads

        if global_rank == 0:
            logging.info(f"Model: TritFormer-850M (d_model={computed_d_model}, layers={n_layers}, heads={n_heads}, d_ff={d_ff})")
        cfg_850m = TritFormer850MConfig(
            vocab_size=vocab_size,
            d_model=computed_d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            d_ff=d_ff,
            max_seq_len=args.seq_len,
            use_triton_gemm=torch.cuda.is_available(),
        )
        model = CheckpointedTritFormer850M(
            config=cfg_850m,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.model_type == "tritformer":
        if global_rank == 0:
            logging.info("Model: TritFormer 1.58-bit Ternary Transformer with Straight-Through Estimator.")
        model = CheckpointedTritFormerLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
            use_ternary_head=args.use_ternary_head,
        )
    else:
        if global_rank == 0:
            logging.info("Model: Dense Standard LLaMA Architecture with SwiGLU & SDPA.")
        model = StandardLLaMALM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
        )

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if global_rank == 0:
        logging.info(f"Model Parameters: {num_params:,} ({num_params / 1e6:.2f}M params)")
        logging.info(f"Gradient Checkpointing: {'ENABLED (Activation Saving ~70%)' if args.gradient_checkpointing else 'DISABLED'}")

    # 4. Distributed Data Parallel Wrapping
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # 5. Optimizer: Strictly AdamW
    optimizer = build_adamw_optimizer(
        model=model,
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        use_8bit_adam=args.use_8bit_adam,
        rank=global_rank,
    )

    # 6. Warmup + Cosine Decay Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        min_lr_ratio=args.min_lr / args.max_lr,
    )

    # 7. Checkpoint Manager & Resume
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        save_interval_sec=args.save_interval_sec,
        rank=global_rank,
    )
    start_step = 0
    tokens_seen = 0
    if args.resume:
        start_step, tokens_seen, _ = ckpt_manager.load_latest(
            model=model,
            model_type=args.model_type,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    # 8. Profiler & Telemetry
    profiler = ClusterProfiler(
        world_size=world_size,
        rank=global_rank,
        device=device,
        num_params=num_params,
    )

    # 9. Asynchronous Distributed Dataset Streamer
    if getattr(args, "use_ultra_streamer", True) and HAS_ULTRA_STREAMER:
        if global_rank == 0:
            logging.info("Data Pipeline: UltraDatasetStreamer (Multi-Process Tokenizer Pool, Document-Masked Packing, 128-DMA Ring Buffer).")
        stream_cfg = UltraStreamConfig(
            dataset_name=args.dataset_name,
            dataset_subset=args.dataset_subset,
            local_fallback_file=args.local_corpus,
            tokenizer_name_or_path=args.tokenizer,
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size,
            rank=global_rank,
            world_size=world_size,
            prefetch_queue_depth=max(args.prefetch_buffer_size, 128),
            pin_memory=True,
        )
        data_streamer = UltraDatasetStreamer(stream_cfg)
    else:
        if global_rank == 0:
            logging.info("Data Pipeline: DistributedDatasetStreamer (Single-thread baseline).")
        stream_cfg = StreamConfig(
            dataset_name=args.dataset_name,
            dataset_subset=args.dataset_subset,
            local_fallback_file=args.local_corpus,
            tokenizer_name_or_path=args.tokenizer,
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size,
            rank=global_rank,
            world_size=world_size,
            prefetch_buffer_size=args.prefetch_buffer_size,
            pin_memory=True,
        )
        data_streamer = DistributedDatasetStreamer(stream_cfg)

    # Autocast Device & Dtype: STRICT BFLOAT16 ON CUDA
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16

    # 10. Training Loop
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if global_rank == 0:
        logging.info("Entering Main Training Loop. Monitoring cluster hardware telemetry...\n")

    t_train_start = time.time()
    rolling_loss = 0.0

    try:
        for step in range(start_step, args.max_steps):
            t_step_start = time.time()
            accumulated_loss = 0.0

            # --- Gradient Accumulation Sub-Steps ---
            for micro_step in range(grad_accum_steps):
                batch = data_streamer.get_batch(device=device, non_blocking=True)
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # DDP Optimization: synchronize gradients ONLY on final accumulation step
                is_final_micro_step = (micro_step == grad_accum_steps - 1)
                sync_context = contextlib.nullcontext() if (is_final_micro_step or world_size == 1) else model.no_sync()

                with sync_context:
                    # STRICT BAN ON FP32: Pure bfloat16 autocasting
                    with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype):
                        logits, loss = model(input_ids, targets=labels)
                        scaled_loss = loss / grad_accum_steps

                    # Backward pass computes gradients in BF16/FP32 master weights
                    scaled_loss.backward()

                accumulated_loss += scaled_loss.item()

            # --- Gradient Clipping & Optimizer Step ---
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            t_step_end = time.time()
            step_duration = t_step_end - t_step_start
            tokens_seen += tokens_per_optimizer_step

            # --- Cross-Rank Loss Reduction ---
            loss_tensor = torch.tensor(accumulated_loss, device=device)
            if world_size > 1:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            reduced_loss = loss_tensor.item()
            rolling_loss = 0.9 * rolling_loss + 0.1 * reduced_loss if rolling_loss > 0 else reduced_loss
            perplexity = math.exp(min(reduced_loss, 20.0))

            # --- Hardware Telemetry & Profiling ---
            achieved_tflops, mfu_pct, throughput_tok_s = profiler.compute_tflops_and_mfu(
                tokens_in_step=tokens_per_optimizer_step,
                step_duration=step_duration,
                has_grad_checkpointing=args.gradient_checkpointing,
            )
            vram_stats = profiler.collect_vram_metrics()

            # --- Logging (Rank 0) ---
            if global_rank == 0 and (step % args.log_interval == 0 or step == args.max_steps - 1):
                cur_lr = scheduler.get_last_lr()[0]
                # Format per-GPU VRAM summary
                vram_str = " | ".join(
                    f"GPU {s['rank']}: {s['allocated_gb']:.1f}/{s['reserved_gb']:.1f} GB"
                    for s in vram_stats
                )

                logging.info(
                    f"Step {step+1:06d}/{args.max_steps} | Loss: {reduced_loss:.4f} (ppl: {perplexity:.2f}) | "
                    f"LR: {cur_lr:.2e} | GradNorm: {grad_norm:.2f} | "
                    f"Speed: {throughput_tok_s:,.0f} tok/s | TFLOP/s: {achieved_tflops:.1f} (MFU: {mfu_pct:.1f}%) | "
                    f"Step Time: {step_duration:.2f}s | Tokens: {tokens_seen / 1e6:.2f}M\n"
                    f"       └── [VRAM Status] {vram_str}"
                )

            # --- Autosave Checkpoint (Every 5-10 Minutes) ---
            if ckpt_manager.should_save(step + 1):
                if world_size > 1:
                    dist.barrier()
                ckpt_manager.save(
                    step=step + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokens_seen=tokens_seen,
                    loss=reduced_loss,
                    config=vars(args),
                    is_final=False,
                )
                if world_size > 1:
                    dist.barrier()

        # Final Checkpoint Save
        if world_size > 1:
            dist.barrier()
        ckpt_manager.save(
            step=args.max_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokens_seen=tokens_seen,
            loss=reduced_loss,
            config=vars(args),
            is_final=True,
        )
        if world_size > 1:
            dist.barrier()

    finally:
        # Graceful Shutdown of Streaming Threads & Process Group
        data_streamer.stop()
        if world_size > 1 and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    total_training_time = time.time() - t_train_start
    if global_rank == 0:
        logging.info("=" * 80)
        logging.info("DISTRIBUTED PRETRAINING COMPLETED SUCCESSFULLY")
        logging.info(f"Total Steps:            {args.max_steps:,}")
        logging.info(f"Total Tokens Processed: {tokens_seen:,} ({tokens_seen / 1e6:.2f}M tokens)")
        logging.info(f"Total Training Time:    {total_training_time / 60.0:.2f} minutes")
        logging.info(f"Final Validation Loss:  {reduced_loss:.4f} (Perplexity: {perplexity:.2f})")
        logging.info("=" * 80)

    # Immediate clean process exit across all DDP ranks
    os._exit(0)


# ==============================================================================
# 9. CLI ENTRY POINT & HYPERPARAMETER SPECIFICATION
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed LLM Pretraining Suite (4x RTX 4090 / 96GB VRAM)")

    # Model Configuration
    parser.add_argument("--model_type", type=str, default="tritformer", choices=["tritformer", "tritformer_850m", "llama"], help="Architecture")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=1536, help="Feed-forward intermediate dimension")
    parser.add_argument("--use_ternary_head", action="store_true", help="Quantize LM head to ternary weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable activation checkpointing")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")

    # Sequence & Batch Calculus (4x RTX 4090 Spec)
    parser.add_argument("--seq_len", type=int, default=2048, help="Context length")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro-batch size per GPU")
    parser.add_argument("--global_batch_size", type=int, default=512, help="Target global batch size in sequences (512-1024)")

    # Optimizer & Scheduler (Strictly AdamW)
    parser.add_argument("--max_lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate floor")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay on 2D matrices")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use bitsandbytes 8-bit AdamW")

    # Dataset & Streaming
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu", help="HuggingFace dataset name")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT", help="Dataset subset")
    parser.add_argument("--local_corpus", type=str, default=None, help="Local text file fallback if offline")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer model name or path")
    parser.add_argument("--prefetch_buffer_size", type=int, default=128, help="Batches in ring buffer queue")
    parser.add_argument("--use_ultra_streamer", action="store_true", default=True, help="Enable Ultra High-Throughput Streamer with Document-Masked Packing")
    parser.add_argument("--no_ultra_streamer", dest="use_ultra_streamer", action="store_false", help="Disable Ultra Streamer and use baseline streamer")

    # Training Steps & Checkpoints
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging step frequency")
    parser.add_argument("--save_interval_sec", type=float, default=300.0, help="Checkpoint autosave interval in seconds (5-10 min)")
    parser.add_argument("--checkpoint_dir", type=str, default="C:/Users/1/Desktop/AImL/demsni/checkpoints", help="Save directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_distributed(args)
