#!/usr/bin/env bash
# Shell Distributed Training Launcher for 4x NVIDIA RTX 4090 (96GB VRAM)
# Target: C:\Users\1\Desktop\AImL\demsni\run_4x4090.sh

set -e

echo "================================================================================"
echo "LAUNCHING DISTRIBUTED TRAINING: 4x RTX 4090 CLUSTER (96GB VRAM TOTAL)"
echo "Model: TritFormer 1.58-bit Ternary LLM | Context: 2048 | Target Global Batch: 512"
echo "================================================================================"

export CUDA_DEVICE_MAX_CONNECTIONS="1"
export NCCL_DEBUG="INFO"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):$PYTHONPATH"

# Run torchrun with 4 processes (1 process per RTX 4090 GPU)
torchrun --nproc_per_node=4 \
    "$(dirname "${BASH_SOURCE[0]}")/train_distributed.py" \
    --model_type tritformer_850m \
    --seq_len 2048 \
    --micro_batch_size 4 \
    --global_batch_size 512 \
    --gradient_checkpointing \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 2000 \
    --max_steps 50000 \
    --dataset_name "HuggingFaceFW/fineweb-edu" \
    --dataset_subset "sample-10BT" \
    --tokenizer "gpt2" \
    --prefetch_buffer_size 64 \
    --save_interval_sec 300 \
    --checkpoint_dir "$(dirname "${BASH_SOURCE[0]}")/checkpoints" \
    "$@"
