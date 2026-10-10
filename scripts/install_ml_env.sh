#!/usr/bin/env bash
set -e

VENV_PATH="${HOME}/venv_4090"
if [ ! -d "${VENV_PATH}" ]; then
    python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

pip install --upgrade pip setuptools wheel ninja packaging

echo "=== Installing PyTorch with CUDA 12.4 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing Core ML Libraries ==="
pip install transformers datasets accelerate bitsandbytes triton tiktoken wandb rich psutil scipy sentencepiece tokenizers einops safetensors

echo "=== Testing 4x GPU PyTorch ==="
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count(), 'BF16:', torch.cuda.is_bf16_supported())"

echo "=== SUCCESS ==="
