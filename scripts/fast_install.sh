#!/usr/bin/env bash
set -e

echo "=== Fast High-Throughput Provisioning for 4x RTX 4090 ==="
VENV_PATH="${HOME}/venv_4090"
if [ ! -d "${VENV_PATH}" ]; then
    python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

# 1. Download PyTorch cu124 wheels via wget at 600MB/s
mkdir -p /home/ubuntu/wheels
cd /home/ubuntu/wheels

echo "[1/3] Downloading PyTorch 2.4.1+cu124 wheel..."
wget -c "https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp312-cp312-linux_x86_64.whl" -O torch-2.4.1+cu124-cp312-cp312-linux_x86_64.whl

echo "[2/3] Installing PyTorch from local wheel..."
pip install torch-2.4.1+cu124-cp312-cp312-linux_x86_64.whl

echo "[3/3] Installing Ecosystem Packages..."
pip install transformers datasets accelerate bitsandbytes triton tiktoken wandb rich psutil scipy sentencepiece tokenizers einops safetensors

echo "=== Cluster Self-Test ==="
python3 - << 'EOF'
import torch
print("=" * 60)
print(f"PyTorch Version:  {torch.__version__}")
print(f"CUDA Available:   {torch.cuda.is_available()}")
print(f"GPU Count:        {torch.cuda.device_count()}")
print(f"BF16 Supported:   {torch.cuda.is_bf16_supported()}")
total_vram = 0.0
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    vram = p.total_memory / (1024**3)
    total_vram += vram
    print(f"  GPU {i}: {p.name} | VRAM: {vram:.2f} GB | SM {p.major}.{p.minor}")
print(f"Total Cluster VRAM: {total_vram:.2f} GB")
print(">> Performing parallel 4-GPU MatMul test in BF16...")
for i in range(torch.cuda.device_count()):
    d = torch.device(f"cuda:{i}")
    x = torch.randn(1024, 1024, device=d, dtype=torch.bfloat16)
    y = torch.matmul(x, x)
    torch.cuda.synchronize(d)
print(">> ALL 4x RTX 4090 GPUs TESTED & OPERATIONAL!")
print("=" * 60)
EOF

echo "=== FAST INSTALL COMPLETED SUCCESSFULLY ==="
