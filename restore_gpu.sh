#!/bin/bash
#
# restore_gpu.sh - Restore GPU settings after tuning
#
# This script reads gpu_settings_backup.txt and restores GPU to original state
#

set -e

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

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    log_error "Please run with sudo: sudo ./restore_gpu.sh"
    exit 1
fi

SETTINGS_FILE="gpu_settings_backup.txt"

# Check if settings file exists
if [ ! -f "$SETTINGS_FILE" ]; then
    log_error "Settings file '$SETTINGS_FILE' not found!"
    log_error "Cannot restore GPU settings. You may need to reset manually:"
    echo ""
    echo "  sudo nvidia-smi -rgc      # Reset GPU clock"
    echo "  sudo nvidia-smi -rmc      # Reset memory clock"
    echo "  sudo nvidia-smi -rpl      # Reset power limit"
    echo "  sudo nvidia-smi -pm 0     # Disable persistence mode"
    echo ""
    exit 1
fi

log_info "Restoring GPU settings from ${SETTINGS_FILE}..."
echo "=================================================="

# Read settings from file
source ${SETTINGS_FILE}

log_info "GPU ID: ${GPU_ID}"
log_info "Original persistence mode: ${ORIGINAL_PERSISTENCE}"
log_info "Original power limit: ${ORIGINAL_POWER}W"

echo "=================================================="

# Reset GPU clocks
log_info "Resetting GPU clocks..."
nvidia-smi -i ${GPU_ID} -rgc
log_info "✓ GPU clock reset to default"

# Reset memory clocks
log_info "Resetting memory clocks..."
nvidia-smi -i ${GPU_ID} -rmc
log_info "✓ Memory clock reset to default"

# Restore power limit
log_info "Restoring power limit to ${ORIGINAL_POWER}W..."
nvidia-smi -i ${GPU_ID} -pl ${ORIGINAL_POWER}
log_info "✓ Power limit restored"

# Restore persistence mode
if [ "$ORIGINAL_PERSISTENCE" = "Disabled" ]; then
    log_info "Disabling persistence mode..."
    nvidia-smi -i ${GPU_ID} -pm 0
    log_info "✓ Persistence mode disabled"
else
    log_info "Keeping persistence mode enabled (was already enabled)"
fi

echo "=================================================="
log_info "GPU settings restored successfully!"
echo "=================================================="

# Show current GPU status
nvidia-smi -i ${GPU_ID}

# Clean up settings file
log_info "Removing backup file ${SETTINGS_FILE}..."
rm -f ${SETTINGS_FILE}
log_info "✓ Backup file removed"

echo ""
log_info "GPU restoration complete!"
