#!/bin/bash
#
# runtuning.sh - GPU Tuning Script with Manual Settings Management
#
# This script:
# 1. Saves current GPU settings to a file
# 2. Sets GPU to max performance (persistence mode, max clocks)
# 3. Runs the TVM Conv2D tuning
# 4. YOU MUST RUN restore_gpu.sh AFTERWARD to restore settings
#

set -ex  # Exit on error

# Configuration
GPU_ID=0
SCRIPT_NAME="batch_conv2d_cuda_tuning.py"
SETTINGS_FILE="gpu_settings_backup.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    log_error "Please run with sudo: sudo ./runtuning.sh"
    exit 1
fi

log_info "Starting GPU tuning setup..."
echo "=================================================="

# Save current GPU settings to file
log_info "Saving current GPU settings to ${SETTINGS_FILE}..."
ORIGINAL_PERSISTENCE=$(nvidia-smi -i ${GPU_ID} --query-gpu=persistence_mode --format=csv,noheader)
ORIGINAL_POWER=$(nvidia-smi -i ${GPU_ID} --query-gpu=power.limit --format=csv,noheader,nounits)

echo "GPU_ID=${GPU_ID}" > ${SETTINGS_FILE}
echo "ORIGINAL_PERSISTENCE=${ORIGINAL_PERSISTENCE}" >> ${SETTINGS_FILE}
echo "ORIGINAL_POWER=${ORIGINAL_POWER}" >> ${SETTINGS_FILE}

log_info "Current persistence mode: ${ORIGINAL_PERSISTENCE}"
log_info "Current power limit: ${ORIGINAL_POWER}W"

# Get max supported clocks
log_info "Detecting maximum GPU clocks..."
MAX_GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} -q -d SUPPORTED_CLOCKS | grep "Graphics" | head -1 | awk '{print $3}')
MAX_MEM_CLOCK=$(nvidia-smi -i ${GPU_ID} -q -d SUPPORTED_CLOCKS | grep "Memory" | head -1 | awk '{print $3}')

if [ -z "$MAX_GPU_CLOCK" ] || [ -z "$MAX_MEM_CLOCK" ]; then
    log_error "Failed to detect GPU clocks. Please check GPU availability."
    exit 1
fi

log_info "Maximum GPU clock: ${MAX_GPU_CLOCK} MHz"
log_info "Maximum memory clock: ${MAX_MEM_CLOCK} MHz"

# Get max power limit
MAX_POWER=$(nvidia-smi -i ${GPU_ID} -q -d POWER | grep "Max Power Limit" | awk '{print $5}')
log_info "Maximum power limit: ${MAX_POWER}W"

echo "=================================================="

# Set GPU to maximum performance
log_info "Configuring GPU for benchmarking..."
echo "=================================================="

# Enable persistence mode
log_info "Enabling persistence mode..."
nvidia-smi -i ${GPU_ID} -pm 1
log_info "✓ Persistence mode enabled"

# Set maximum power limit
log_info "Setting power limit to ${MAX_POWER}W..."
nvidia-smi -i ${GPU_ID} -pl ${MAX_POWER}
log_info "✓ Power limit set to ${MAX_POWER}W"

# Lock GPU clocks to maximum
log_info "Locking GPU clock to ${MAX_GPU_CLOCK} MHz..."
nvidia-smi -i ${GPU_ID} -lgc ${MAX_GPU_CLOCK},${MAX_GPU_CLOCK}
log_info "✓ GPU clock locked to ${MAX_GPU_CLOCK} MHz"

# Lock memory clocks to maximum
log_info "Locking memory clock to ${MAX_MEM_CLOCK} MHz..."
nvidia-smi -i ${GPU_ID} -lmc ${MAX_MEM_CLOCK},${MAX_MEM_CLOCK}
log_info "✓ Memory clock locked to ${MAX_MEM_CLOCK} MHz"

echo "=================================================="
log_info "GPU configuration complete. Current status:"
echo "=================================================="

# Display GPU status
nvidia-smi -i ${GPU_ID}

echo "=================================================="
log_info "Starting TVM Conv2D tuning..."
echo "=================================================="

exit 0

# Run the tuning script (as the original user, not root)
ORIGINAL_USER=$(logname 2>/dev/null || echo $SUDO_USER)
if [ -n "$ORIGINAL_USER" ]; then
    log_info "Running as user: ${ORIGINAL_USER}"
    su - ${ORIGINAL_USER} -c "cd $(pwd) && python ${SCRIPT_NAME}"
else
    log_warn "Could not detect original user, running as root"
    python ${SCRIPT_NAME}
fi

TUNING_EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $TUNING_EXIT_CODE -eq 0 ]; then
    log_info "Tuning completed successfully!"
else
    log_error "Tuning failed with exit code ${TUNING_EXIT_CODE}"
fi
echo "=================================================="
echo ""
log_warn "IMPORTANT: Run 'sudo ./restore_gpu.sh' to restore GPU settings!"
echo ""

exit $TUNING_EXIT_CODE
