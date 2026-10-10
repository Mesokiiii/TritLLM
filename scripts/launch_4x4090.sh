#!/usr/bin/env bash
# ==============================================================================
# TritFormer Production Launch Script: 4x NVIDIA GeForce RTX 4090
# Node ID: 4f0ffad9-f728-41ca-8a6a-2704bdfad4e1
# Target: Ubuntu 24.04 CUDA | 4x RTX 4090 (96GB VRAM)
# ==============================================================================

set -euo pipefail

# ANSI Color codes
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "================================================================================"
echo "    LAUNCHING DISTRIBUTED TRAINING ON 4x NVIDIA GEFORCE RTX 4090               "
echo "    Architecture: TritFormer LM / DistributedDataParallel (DDP)                "
echo "    Total GPUs: 4 | Global Batch: 512 (SeqLen: 2048, ~1.05M tokens/step)       "
echo "================================================================================"
echo -e "${RESET}"

# ------------------------------------------------------------------------------
# 1. Environment & Python Virtualenv Verification
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_PATH="${HOME}/venv_4090"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "${VENV_PATH}/bin/activate" ]; then
        echo -e "${YELLOW}[*] Activating virtual environment from ${VENV_PATH}...${RESET}"
        # shellcheck source=/dev/null
        source "${VENV_PATH}/bin/activate"
    elif [ -f "./venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "./venv/bin/activate"
    fi
fi

# ------------------------------------------------------------------------------
# 2. Multi-GPU Distributed Environment Variables
# ------------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# CPU Threading (16 vCPUs total / 4 GPUs = 4 threads per rank)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PyTorch & NCCL Optimization Flags for RTX 4090 PCIe Interconnect
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1

# PCIe tuning for consumer 4090 GPUs (no NVLink, PCIe Gen4 x16/x8)
# Disables InfiniBand search (single-node) to avoid delays
export NCCL_IB_DISABLE=1

# Set CUDA allocator configuration to avoid memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo -e "${GREEN}[+] Environment configured:${RESET}"
echo "    CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "    OMP_NUM_THREADS      = ${OMP_NUM_THREADS}"
echo "    NCCL_DEBUG           = ${NCCL_DEBUG}"
echo "    Master Port          = 29500"

# ------------------------------------------------------------------------------
# 3. Create Logging Directories and Telemetry Monitor
# ------------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

DMON_LOG="${LOG_DIR}/gpu_telemetry_${TIMESTAMP}.log"
TRAIN_LOG="${LOG_DIR}/training_${TIMESTAMP}.log"

echo -e "${CYAN}[*] Starting background GPU hardware telemetry monitor...${RESET}"
echo -e "    Telemetry Log: ${DMON_LOG}"
echo -e "    Training Output Log: ${TRAIN_LOG}"

# Background monitoring of GPU power, temp, utilization, VRAM via nvidia-smi dmon
# Flags: -s pucmt (Power, Utilization, Clock, Memory, Temperature) every 2 seconds
nvidia-smi dmon -s pucmt -d 2 > "${DMON_LOG}" 2>&1 &
MONITOR_PID=$!

cleanup() {
    echo -e "\n${YELLOW}[*] Shutting down training and background telemetry monitor (PID: ${MONITOR_PID})...${RESET}"
    if kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
    echo -e "${GREEN}[+] Telemetry saved to ${DMON_LOG}${RESET}"
    echo -e "${GREEN}[+] Training session ended.${RESET}"
}
trap cleanup EXIT SIGINT SIGTERM

# ------------------------------------------------------------------------------
# 4. Distributed Training Launch via torchrun
# ------------------------------------------------------------------------------
echo -e "${BOLD}${GREEN}"
echo "--------------------------------------------------------------------------------"
echo ">> Launching torchrun (4 workers on localhost:29500)..."
echo "--------------------------------------------------------------------------------"
echo -e "${RESET}"

TRAIN_SCRIPT="train_distributed.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}[ERROR] ${TRAIN_SCRIPT} not found in ${SCRIPT_DIR}!${RESET}"
    exit 1
fi

START_TRAIN_TIME=$(date +%s)

# Execute torchrun with full pipe to stdout and tee to log file
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    "${TRAIN_SCRIPT}" \
    --model_type tritformer_850m \
    --micro_batch_size 2 \
    --global_batch_size 64 \
    --seq_len 2048 \
    --gradient_checkpointing \
    --max_steps 100 \
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
    echo -e "${BOLD}${GREEN}>> TRAINING COMPLETED SUCCESSFULLY IN ${TOTAL_TRAIN_SEC} SECONDS! <<${RESET}"
else
    echo -e "${BOLD}${RED}>> TRAINING FINISHED WITH CODE ${TRAIN_EXIT_CODE} AFTER ${TOTAL_TRAIN_SEC}s <<${RESET}"
fi
echo -e "${BOLD}${CYAN}================================================================================${RESET}"

# Summary of GPU Peak Telemetry from monitor log
if [ -f "${DMON_LOG}" ]; then
    echo -e "\n${BOLD}--- GPU Thermal & Utilization Highlights ---${RESET}"
    echo "Recent telemetry samples:"
    tail -n 12 "${DMON_LOG}" || true
fi

exit ${TRAIN_EXIT_CODE}
