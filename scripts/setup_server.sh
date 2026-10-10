#!/usr/bin/env bash
# ==============================================================================
# TritFormer / Deep Learning Multi-GPU Automated Setup Script
# Target: Ubuntu 24.04 LTS (x86_64) | 4x NVIDIA GeForce RTX 4090 (24GB VRAM)
# Server ID: 4f0ffad9-f728-41ca-8a6a-2704bdfad4e1
# Configuration: rtx4090-4.16.128.320 (4x RTX 4090, 16 vCPU, 128GB RAM, 320GB NVMe)
# ==============================================================================

set -euo pipefail

# Color palette for terminal output
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

log_info()  { echo -e "${CYAN}${BOLD}[INFO]${RESET} $1"; }
log_ok()    { echo -e "${GREEN}${BOLD}[OK]${RESET} $1"; }
log_warn()  { echo -e "${YELLOW}${BOLD}[WARN]${RESET} $1"; }
log_error() { echo -e "${RED}${BOLD}[ERROR]${RESET} $1"; }

START_TIME=$(date +%s)

echo -e "${BOLD}${CYAN}"
echo "================================================================================"
echo "    4x RTX 4090 PROVISIONING & ENVIRONMENT SETUP - UBUNTU 24.04 CUDA           "
echo "    Node Config: 4x RTX 4090 (96GB VRAM) | 16 vCPUs | 128GB RAM | 320GB NVMe   "
echo "================================================================================"
echo -e "${RESET}"

# ------------------------------------------------------------------------------
# STEP 1: Hardware & System Resource Verification
# ------------------------------------------------------------------------------
log_info "Step 1/6: Verifying hardware architecture and GPU topology..."

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found! Ensure NVIDIA drivers are installed."
    exit 1
fi

echo -e "\n${BOLD}--- NVIDIA SMI System Overview ---${RESET}"
nvidia-smi
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
log_info "Detected GPU Count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 4 ]; then
    log_warn "Expected 4 GPUs, but found ${GPU_COUNT}. Proceeding with detected GPUs."
else
    log_ok "All 4x NVIDIA GeForce RTX 4090 GPUs are active and recognized."
fi

# Print GPU Model and VRAM details
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pstate --format=csv,noheader

# CPU and RAM Check
CPU_CORES=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
DISK_FREE_GB=$(df -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')
log_ok "System Specs: ${CPU_CORES} vCPUs, ${RAM_GB}GB RAM, ${DISK_FREE_GB}GB NVMe available."

# ------------------------------------------------------------------------------
# STEP 2: System Package Installation
# ------------------------------------------------------------------------------
log_info "Step 2/6: Installing core system dependencies and build utilities..."

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y

sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    rsync \
    tmux \
    htop \
    nvtop \
    iotop \
    pciutils \
    net-tools \
    python3-pip \
    python3-venv \
    python3-dev \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    libssl-dev \
    ca-certificates

log_ok "Core system packages installed successfully."

# ------------------------------------------------------------------------------
# STEP 3: Python Virtual Environment Creation
# ------------------------------------------------------------------------------
VENV_PATH="${HOME}/venv_4090"
log_info "Step 3/6: Setting up Python virtual environment at ${VENV_PATH}..."

if [ ! -d "${VENV_PATH}" ]; then
    python3 -m venv "${VENV_PATH}"
    log_ok "Created clean virtualenv at ${VENV_PATH}."
else
    log_info "Virtual environment already exists at ${VENV_PATH}."
fi

# Activate virtualenv
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

# Ensure ~/.bashrc automatically activates this venv for interactive shells
if ! grep -q "source ${VENV_PATH}/bin/activate" "${HOME}/.bashrc" 2>/dev/null; then
    echo -e "\n# Auto-activate 4x RTX 4090 Python Environment\nsource ${VENV_PATH}/bin/activate" >> "${HOME}/.bashrc"
    log_ok "Added virtualenv auto-activation to ${HOME}/.bashrc."
fi

# Upgrade pip, wheel, setuptools
pip install --upgrade pip setuptools wheel ninja packaging

log_ok "Python environment configured: $(python3 --version), pip $(pip --version | awk '{print $2}')"

# ------------------------------------------------------------------------------
# STEP 4: PyTorch 2.4+ with CUDA 12.4+ Installation
# ------------------------------------------------------------------------------
log_info "Step 4/6: Installing PyTorch 2.4+ with CUDA 12.4 (BF16, FlashAttention, Triton support)..."

# Install latest PyTorch with CUDA 12.4 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

PYTORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
BF16_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_bf16_supported())")

log_ok "PyTorch installed: ${PYTORCH_VER} (CUDA ${CUDA_VER}, BF16 Supported: ${BF16_AVAIL})"

# ------------------------------------------------------------------------------
# STEP 5: FlashAttention-2 & Core ML Stack Installation
# ------------------------------------------------------------------------------
log_info "Step 5/6: Installing FlashAttention-2 and Machine Learning Ecosystem..."

# Set build jobs to prevent OOM during compilation
export MAX_JOBS=8

# Install FlashAttention-2 with no build isolation for maximum CUDA acceleration
log_info "Compiling / installing flash-attn (this may take 2-4 minutes)..."
if pip install flash-attn --no-build-isolation; then
    log_ok "flash-attn installed successfully."
else
    log_warn "Standard flash-attn build failed. Trying with --no-cache-dir..."
    pip install flash-attn --no-build-isolation --no-cache-dir || log_warn "FlashAttention-2 optional install failed. PyTorch SDPA will be used as fallback."
fi

# Install ML libraries
log_info "Installing transformers, datasets, accelerate, bitsandbytes, triton, tiktoken, wandb, rich..."
pip install \
    transformers \
    datasets \
    accelerate \
    bitsandbytes \
    triton \
    tiktoken \
    wandb \
    rich \
    psutil \
    scipy \
    sentencepiece \
    tokenizers \
    einops \
    safetensors

log_ok "Machine learning libraries installed."

# ------------------------------------------------------------------------------
# STEP 6: Multi-GPU Cluster Verification & Self-Test
# ------------------------------------------------------------------------------
log_info "Step 6/6: Running 4x RTX 4090 cluster verification self-test..."

python3 - << 'EOF'
import sys
import torch

print("\n" + "=" * 70)
print(" 4x RTX 4090 HARDWARE & DISTRIBUTED ENVIRONMENT VERIFICATION")
print("=" * 70)

cuda_ok = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"CUDA Available: {cuda_ok}")
print(f"Device Count:   {count} / 4 expected")

if not cuda_ok:
    print("[ERROR] CUDA is not available! Check NVIDIA driver installation.")
    sys.exit(1)

total_vram_gb = 0.0
for i in range(count):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024**3)
    total_vram_gb += vram_gb
    cc = f"{props.major}.{props.minor}"
    print(f"  [GPU {i}] {props.name} | VRAM: {vram_gb:.2f} GB | Compute Capability: {cc}")

print(f"Total Combined VRAM: {total_vram_gb:.2f} GB")

# Check BF16 Support
bf16_supported = torch.cuda.is_bf16_supported()
print(f"BFloat16 Native Acceleration: {'[SUPPORTED]' if bf16_supported else '[NOT SUPPORTED]'}")

# Flash Attention Check
try:
    import flash_attn
    fa_ver = getattr(flash_attn, "__version__", "installed")
    print(f"FlashAttention-2: [AVAILABLE] version {fa_ver}")
except ImportError:
    print("FlashAttention-2: [NOT INSTALLED] (PyTorch SDPA fallback active)")

# Triton Check
try:
    import triton
    print(f"OpenAI Triton: [AVAILABLE] version {triton.__version__}")
except ImportError:
    print("OpenAI Triton: [NOT INSTALLED]")

# Peer-to-Peer Access Matrix across all 4 GPUs
print("\nPCIe P2P Direct Access Matrix:")
header = "     " + " ".join([f"GPU{j}" for j in range(count)])
print(header)
for i in range(count):
    row = [f"GPU{i}"]
    for j in range(count):
        if i == j:
            row.append("  -- ")
        else:
            can_p2p = torch.cuda.can_device_access_peer(i, j)
            row.append(" YES " if can_p2p else "  NO ")
    print(f"{row[0]:4} " + "".join(row[1:]))

# Micro-benchmark: Allocate BF16 tensor on each GPU and perform MatMul
print("\nTesting parallel matrix multiplication across all 4 GPUs...")
try:
    tensors = []
    for i in range(count):
        device = torch.device(f"cuda:{i}")
        a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device)
        tensors.append(c)
    print(">> All 4 GPUs executed BF16 MatMul successfully!")
except Exception as e:
    print(f"[FAIL] GPU MatMul test failed: {e}")
    sys.exit(1)

print("=" * 70)
print(">> CLUSTER STATUS: FULLY OPERATIONAL AND READY FOR DISTRIBUTED TRAINING! <<")
print("=" * 70 + "\n")
EOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "${GREEN}${BOLD}"
echo "================================================================================"
echo "    SERVER PROVISIONING COMPLETE! (Duration: ${DURATION}s)                     "
echo "    To launch 4x RTX 4090 training run:                                        "
echo "      source ${VENV_PATH}/bin/activate                                         "
echo "      bash launch_4x4090.sh                                                    "
echo "================================================================================"
echo -e "${RESET}"
