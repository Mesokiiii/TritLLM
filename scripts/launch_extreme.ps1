# PowerShell Distributed Training Launcher for 4x NVIDIA RTX 4090
# Target: C:\Users\1\Desktop\AImL\demsni\launch_extreme.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "TRITFORMER EXTREME DISTRIBUTED ENGINE (4x RTX 4090 / 96GB VRAM)" -ForegroundColor Green
Write-Host "Architecture: SubLN + Continuous Head | Optimizer: Hybrid Muon+AdamW | WSD Sched" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:CUDA_DEVICE_MAX_CONNECTIONS = "1"
$env:NCCL_DEBUG = "INFO"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$checkpointDir = Join-Path $scriptDir "checkpoints"

torchrun --nproc_per_node=4 `
    (Join-Path $scriptDir "train_extreme.py") `
    --model_type tritformer_extreme `
    --d_model 2048 `
    --n_heads 16 `
    --n_layers 16 `
    --d_ff 5632 `
    --seq_len 2048 `
    --micro_batch_size 4 `
    --global_batch_size 512 `
    --optimizer hybrid_muon `
    --muon_lr 0.02 `
    --adam_lr 0.0006 `
    --warmup_ratio 0.05 `
    --decay_ratio 0.15 `
    --min_lr_ratio 0.1 `
    --compile `
    --compile_mode reduce-overhead `
    --gradient_checkpointing `
    --bucket_cap_mb 25 `
    --eval_interval 20 `
    --eval_batches 8 `
    --repetition_penalty 1.25 `
    --temperature 0.8 `
    --top_p 0.9 `
    --generate_interval 50 `
    --max_steps 500 `
    --log_interval 2 `
    --save_interval_sec 300 `
    --dataset_name "HuggingFaceFW/fineweb-edu" `
    --dataset_subset "sample-10BT" `
    --checkpoint_dir $checkpointDir `
    $args
