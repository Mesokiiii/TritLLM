# PowerShell Distributed Training Launcher for 4x NVIDIA RTX 4090 (96GB VRAM)
# Target: C:\Users\1\Desktop\AImL\demsni\run_4x4090.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "LAUNCHING DISTRIBUTED TRAINING: 4x RTX 4090 CLUSTER (96GB VRAM TOTAL)" -ForegroundColor Green
Write-Host "Model: TritFormer 1.58-bit Ternary LLM | Context: 2048 | Target Global Batch: 512" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

$env:CUDA_DEVICE_MAX_CONNECTIONS = "1"
$env:NCCL_DEBUG = "INFO"
$env:PYTHONPATH = "C:\Users\1\Desktop\AImL;$env:PYTHONPATH"

# Run torchrun with 4 processes (1 process per RTX 4090 GPU)
torchrun --nproc_per_node=4 `
    C:\Users\1\Desktop\AImL\demsni\train_distributed.py `
    --model_type tritformer `
    --d_model 1024 `
    --n_heads 16 `
    --n_layers 16 `
    --d_ff 2816 `
    --seq_len 2048 `
    --micro_batch_size 4 `
    --global_batch_size 512 `
    --gradient_checkpointing `
    --max_lr 6e-4 `
    --min_lr 6e-5 `
    --warmup_steps 2000 `
    --max_steps 50000 `
    --dataset_name "HuggingFaceFW/fineweb-edu" `
    --dataset_subset "sample-10BT" `
    --tokenizer "gpt2" `
    --prefetch_buffer_size 64 `
    --save_interval_sec 300 `
    --checkpoint_dir "C:\Users\1\Desktop\AImL\demsni\checkpoints" `
    $args
