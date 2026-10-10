#!/usr/bin/env bash
# ==============================================================================
# TritFormer Extreme Launch Harness: 4x NVIDIA GeForce RTX 4090
# Hardware: 4x RTX 4090 (96GB VRAM total) | PCIe Gen 4.0 x16/x8 | Ada Lovelace sm_89
# System Optimizations: SubLN, Continuous LM Head, Hybrid Muon+AdamW, WSD Scheduler,
#                       torch.compile(reduce-overhead) + CUDAGraphs, PCIe Overlap
# ==============================================================================

set -euo pipefail

# ANSI Color formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "================================================================================"
echo "    TRITFORMER EXTREME DISTRIBUTED ENGINE: 4x RTX 4090 (96GB VRAM)              "
echo "    Architecture: SubLN + Continuous Head + 1.58-bit Ternary STE                "
echo "    Optimizer:    Hybrid Muon (2D Internal Matrices) + Fused AdamW              "
echo "    Scheduler:    Warmup-Stable-Decay (WSD 5% / 80% / 15%)                      "
echo "    Compilation:  torch.compile(mode='reduce-overhead') + CUDAGraphs            "
echo "    PCIe Tuning:  Async Gradient Accumulation Overlap (model.no_sync)           "
echo "    Validation:   FineWeb-Edu Perplexity Contour every 20 steps                 "
echo "    Inference:    Repetition Penalty (1.25) + Top-P (0.90) + Anti-Loop          "
echo "================================================================================"
echo -e "${RESET}"

# ------------------------------------------------------------------------------
# 1. Directory Setup & Environment Activation
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/train_extreme.py" ]; then
    PROJECT_DIR="${SCRIPT_DIR}"
else
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_DIR}"

VENV_PATH="${HOME}/venv_4090"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "${VENV_PATH}/bin/activate" ]; then
        echo -e "${YELLOW}[*] Activating dedicated Python environment from ${VENV_PATH}...${RESET}"
        # shellcheck source=/dev/null
        source "${VENV_PATH}/bin/activate"
    elif [ -f "./venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "./venv/bin/activate"
    fi
fi

# ------------------------------------------------------------------------------
# 2. Multi-GPU PCIe & NCCL Optimization Flags
# ------------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# CPU Threading (16 vCPUs total / 4 GPUs = 4 threads per rank)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PyTorch & NCCL Communication Flags for consumer PCIe interconnect
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1          # Single-node PCIe; disable InfiniBand search overhead
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Memory Allocator: Avoid fragmentation on large micro-batches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Inductor Compilation Flags for Ada Lovelace (sm_89)
export TORCH_INDUCTOR_COMPILE_THREADS=4
export TORCH_CUDA_ARCH_LIST="8.9"

echo -e "${GREEN}[+] Extreme Cluster Environment Configured:${RESET}"
echo "    CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES}"
echo "    OMP_NUM_THREADS       = ${OMP_NUM_THREADS}"
echo "    TORCH_CUDA_ARCH_LIST  = ${TORCH_CUDA_ARCH_LIST}"
echo "    PYTORCH_CUDA_ALLOC    = ${PYTORCH_CUDA_ALLOC_CONF}"
echo "    NCCL Transport        = PCIe Gen 4 P2P (NCCL_IB_DISABLE=1)"
echo "    Master Port           = 29500"

# ------------------------------------------------------------------------------
# 3. Telemetry Monitor & Logging Directories
# ------------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

DMON_LOG="${LOG_DIR}/gpu_telemetry_${TIMESTAMP}.log"
TRAIN_LOG="${LOG_DIR}/training_extreme_${TIMESTAMP}.log"

echo -e "${CYAN}[*] Initializing background GPU hardware harness monitor...${RESET}"
echo -e "    Telemetry Log:  ${DMON_LOG}"
echo -e "    Training Log:   ${TRAIN_LOG}"

# Background monitoring of GPU power, temp, clocks, VRAM via nvidia-smi dmon
nvidia-smi dmon -s pucmt -d 2 > "${DMON_LOG}" 2>&1 &
MONITOR_PID=$!

cleanup() {
    echo -e "\n${YELLOW}[*] Shutting down training session and background monitor (PID: ${MONITOR_PID})...${RESET}"
    if kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
    echo -e "${GREEN}[+] Telemetry saved to ${DMON_LOG}${RESET}"
    echo -e "${GREEN}[+] Session ended.${RESET}"
}
trap cleanup EXIT SIGINT SIGTERM

# ------------------------------------------------------------------------------
# 4. Launch Distributed Training Engine via torchrun
# ------------------------------------------------------------------------------
echo -e "${BOLD}${GREEN}"
echo "--------------------------------------------------------------------------------"
echo ">> Launching torchrun (4 DDP processes on localhost:29500)..."
echo "--------------------------------------------------------------------------------"
echo -e "${RESET}"

TRAIN_SCRIPT="train_extreme.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}[ERROR] ${TRAIN_SCRIPT} not found in ${SCRIPT_DIR}!${RESET}"
    exit 1
fi

START_TRAIN_TIME=$(date +%s)

# Execute torchrun with full pipe to stdout and tee to training log
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    "${TRAIN_SCRIPT}" \
    --model_type tritformer_extreme \
    --d_model 2048 \
    --n_heads 16 \
    --n_layers 16 \
    --d_ff 5632 \
    --seq_len 2048 \
    --micro_batch_size 4 \
    --global_batch_size 512 \
    --optimizer hybrid_muon \
    --muon_lr 0.02 \
    --adam_lr 0.0006 \
    --muon_momentum 0.95 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --decay_ratio 0.15 \
    --min_lr_ratio 0.1 \
    --compile \
    --compile_mode reduce-overhead \
    --gradient_checkpointing \
    --bucket_cap_mb 25 \
    --eval_interval 20 \
    --eval_batches 8 \
    --repetition_penalty 1.25 \
    --temperature 0.8 \
    --top_p 0.9 \
    --generate_interval 50 \
    --max_steps 500 \
    --log_interval 2 \
    --save_interval_sec 300 \
    --dataset_name "HuggingFaceFW/fineweb-edu" \
    --dataset_subset "sample-10BT" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    "$@" 2>&1 | tee "${TRAIN_LOG}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
END_TRAIN_TIME=$(date +%s)
TOTAL_TRAIN_SEC=$((END_TRAIN_TIME - START_TRAIN_TIME))

echo ""
echo -e "${BOLD}${CYAN}================================================================================${RESET}"
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo -e "${BOLD}${GREEN}>> EXTREME TRAINING RUN COMPLETED SUCCESSFULLY IN ${TOTAL_TRAIN_SEC} SECONDS! <<${RESET}"
else
    echo -e "${BOLD}${RED}>> TRAINING FINISHED WITH CODE ${TRAIN_EXIT_CODE} AFTER ${TOTAL_TRAIN_SEC}s <<${RESET}"
fi
echo -e "${BOLD}${CYAN}================================================================================${RESET}"

# Summary of GPU Peak Telemetry from monitor log
if [ -f "${DMON_LOG}" ]; then
    echo -e "\n${BOLD}--- GPU Hardware Highlights (Last Samples) ---${RESET}"
    tail -n 12 "${DMON_LOG}" || true
fi

exit ${TRAIN_EXIT_CODE}
