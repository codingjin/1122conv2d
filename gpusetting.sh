#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <mode_number>"
    echo "  mode 1: power max, clock max"
    echo "  mode 2: power max, clock min"
    echo "  mode 3: power min, clock max"
    echo "  mode 4: power min, clock min"
    exit 1
fi

MODE=$1
GPU_ID=0
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPU(s)."

sudo nvidia-smi -i ${GPU_ID} -pm 1

# ------------------------------------------------------------
# Get clean MIN/MAX power limits
# ------------------------------------------------------------
read MIN_POWER MAX_POWER < <(nvidia-smi -i ${GPU_ID} -q -d POWER \
  | awk '/GPU Power Readings/,/Power Samples/ {if ($1=="Min") p1=$5; if ($1=="Max") p2=$5} END{print p1, p2}')

if [[ -z "$MIN_POWER" || -z "$MAX_POWER" ]]; then
    echo "❌ Failed to parse power limits from nvidia-smi"
    exit 1
fi

if [[ "$MODE" == "1" || "$MODE" == "2" ]]; then
    POWER_LIMIT=$MAX_POWER
else
    POWER_LIMIT=$MIN_POWER
fi

echo "→ Setting GPU${GPU_ID} power limit to ${POWER_LIMIT} W"
sudo nvidia-smi -i ${GPU_ID} -pl ${POWER_LIMIT}

# ------------------------------------------------------------
# Determine GPU core clock only (no memory)
# ------------------------------------------------------------
if [[ "$MODE" == "1" || "$MODE" == "3" ]]; then
    # Max GPU clock
    GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
        | tr -d ' ' | sort -nr -t',' -k2 | head -n1 | awk -F',' '{print $2}' | grep -Eo '[0-9]+')
else
    # Min GPU clock
    GPU_CLOCK=$(nvidia-smi -i ${GPU_ID} --query-supported-clocks=memory,graphics --format=csv,noheader \
        | tr -d ' ' | sort -n -t',' -k2 | head -n1 | awk -F',' '{print $2}' | grep -Eo '[0-9]+')
fi

echo "→ Locking GPU core clock to ${GPU_CLOCK} MHz"
sudo nvidia-smi -i ${GPU_ID} --lock-gpu-clocks=${GPU_CLOCK},${GPU_CLOCK}

# ------------------------------------------------------------
# Disable other GPUs
# ------------------------------------------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    echo ""
    echo "Disabling other GPUs..."
    for ((i=1; i<$NUM_GPUS; i++)); do
        BUS_ID=$(nvidia-smi -i $i -q | grep "Bus Id" | awk '{print $4}')
        echo "→ Unbinding GPU $i (Bus $BUS_ID) from driver"
        if [ -n "$BUS_ID" ]; then
            sudo sh -c "echo $BUS_ID > /sys/bus/pci/drivers/nvidia/unbind" || true
        fi
    done
fi

export CUDA_VISIBLE_DEVICES=0
echo ""
echo "CUDA_VISIBLE_DEVICES=0"
echo "✅ GPU0 configured in mode $MODE (GPU clock only); other GPUs disabled."
echo "Check with: nvidia-smi -q -d POWER,CLOCK"
