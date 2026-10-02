#!/bin/bash
# File: launch_worker.sh

# Clean ALL torch extension caches to prevent any compilation
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true
rm -rf /tmp/torch_extensions/ 2>/dev/null || true
rm -rf /var/tmp/torch_extensions/ 2>/dev/null || true

# FSDP optimization settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export PYTHONWARNINGS="ignore::UserWarning"

# Multi-node NCCL settings (from successful test)
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# GPU load balancing settings
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=256

# Load environment using the standard pattern (disable strict mode temporarily for bashrc)
set +u 2>/dev/null || true
source ~/.bashrc && module load GCC/10.2.0 && module load CUDA/12.1.0 && conda activate paramem
set -eo pipefail

cd /home/mmahaut/projects/paramem

# Detect actually available GPUs dynamically
echo "[$(hostname)] SLURM allocated GPUs: ${SLURM_JOB_GPUS:-not set}"
echo "[$(hostname)] Detecting available GPUs..."

# First, get all physically available GPUs using nvidia-smi
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd "," -)
echo "[$(hostname)] Physically available GPUs (nvidia-smi): $AVAILABLE_GPUS"

# If SLURM provided GPU allocation, try to use only those
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    ALLOCATED_GPUS="$SLURM_JOB_GPUS"
    echo "[$(hostname)] SLURM GPU allocation: $ALLOCATED_GPUS"
    
    # Test each SLURM-allocated GPU to see if it's actually available
    HEALTHY_GPUS=""
    IFS=',' read -ra GPU_ARRAY <<< "$ALLOCATED_GPUS"
    for gpu in "${GPU_ARRAY[@]}"; do
        # Check if this GPU exists in the available list
        if echo ",$AVAILABLE_GPUS," | grep -q ",$gpu,"; then
            if CUDA_VISIBLE_DEVICES=$gpu python3 -c "import torch; torch.cuda.init(); assert torch.cuda.device_count() > 0" &>/dev/null; then
                HEALTHY_GPUS="${HEALTHY_GPUS:+$HEALTHY_GPUS,}$gpu"
                echo "[$(hostname)] GPU $gpu: HEALTHY"
            else
                echo "[$(hostname)] GPU $gpu: EXISTS but PyTorch initialization failed (skipping)"
            fi
        else
            echo "[$(hostname)] GPU $gpu: NOT PHYSICALLY PRESENT (skipping)"
        fi
    done
    
    # If no SLURM GPUs were healthy, fall back to all available GPUs
    if [ -z "$HEALTHY_GPUS" ]; then
        echo "[$(hostname)] WARNING: No SLURM-allocated GPUs are healthy!"
        echo "[$(hostname)] Falling back to all physically available GPUs: $AVAILABLE_GPUS"
        HEALTHY_GPUS="$AVAILABLE_GPUS"
    fi
else
    # No SLURM allocation, use all available GPUs
    echo "[$(hostname)] No SLURM GPU allocation, using all available GPUs"
    HEALTHY_GPUS="$AVAILABLE_GPUS"
fi

# Validate we have at least one GPU
if [ -z "$HEALTHY_GPUS" ]; then
    echo "[$(hostname)] ERROR: No healthy GPUs found! Cannot proceed with training."
    nvidia-smi || echo "nvidia-smi failed"
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES to only healthy GPUs
export CUDA_VISIBLE_DEVICES="$HEALTHY_GPUS"
ACTUAL_GPU_COUNT=$(echo "$HEALTHY_GPUS" | tr ',' '\n' | grep -c .)
EXPECTED_GPUS_PER_NODE=${GPUS_PER_NODE:-$ACTUAL_GPU_COUNT}

# In multinode runs, all nodes must expose the same nproc_per_node for torchrun.
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ] && [ "$ACTUAL_GPU_COUNT" -ne "$EXPECTED_GPUS_PER_NODE" ]; then
    echo "[$(hostname)] ERROR: expected $EXPECTED_GPUS_PER_NODE healthy GPUs but found $ACTUAL_GPU_COUNT."
    echo "[$(hostname)] Refusing multinode launch to avoid rendezvous/world-size mismatch."
    exit 2
fi

echo "[$(hostname)] Allocated GPUs: ${ALLOCATED_GPUS:-none}"
echo "[$(hostname)] Healthy GPUs: $HEALTHY_GPUS"
echo "[$(hostname)] Actual GPU count: $ACTUAL_GPU_COUNT"
nvidia-smi || echo "Warning: nvidia-smi not available"

# Apply memory optimizations
echo "[$(hostname)] Applying memory optimizations..."
./scripts/entrypoints/maintenance/optimize_memory.sh

echo "[$(hostname)] Running on node rank: $SLURM_NODEID, task: $SLURM_PROCID"
echo "[$(hostname)] Training mode: $([ "$USE_LORA" -eq 1 ] && echo 'LoRA' || echo 'Full fine-tuning')"
echo "[$(hostname)] GPU configuration: ${NUM_GPUS:-1} GPUs per node"

# Memory optimization settings - aggressive memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6,roundup_power2_divisions:16
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
# Additional FSDP optimizations
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

# Clean any problematic cache before starting
echo "[$(hostname)] Thoroughly cleaning ALL extension caches..."
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true
rm -rf /tmp/torch_extensions/ 2>/dev/null || true
find /tmp -name "*torch*" -type d -exec rm -rf {} + 2>/dev/null || true
# pip install deepspeed --force-reinstall
# poetry add deepspeed
# poetry check
# Use strict expected GPU count for multinode, but actual count for single-node.
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    NUM_GPUS_PER_NODE=$EXPECTED_GPUS_PER_NODE
else
    NUM_GPUS_PER_NODE=$ACTUAL_GPU_COUNT
fi
echo "[$(hostname)] Multi-node training: $SLURM_JOB_NUM_NODES nodes, $NUM_GPUS_PER_NODE GPUs per node"
TOTAL_ACTUAL_GPUS=$((SLURM_JOB_NUM_NODES * NUM_GPUS_PER_NODE))
echo "[$(hostname)] Total GPUs in job: $TOTAL_ACTUAL_GPUS"

# Set batch size and grad accumulation based on training type
# Goal: Match effective batch sizes between LoRA and Full FT
# Effective batch size = batch_size × num_gpus × grad_accum
# With 6,170 samples and effective_batch=16: ~386 steps/epoch
# SAVE_STEPS=130 gives ~3 checkpoints/epoch, 15 total checkpoints over 5 epochs
SAVE_STEPS=130

# Check if this is MMLU dataset (reduce batch size for large samples)
IS_MMLU=0
if [[ "$DATASET_NAME" == *"mmlu"* ]]; then
    IS_MMLU=1
    echo "[$(hostname)] Detected MMLU dataset - using reduced batch sizes"
fi

if [ "$USE_LORA" -eq 1 ]; then
    # LoRA: 2 or 4 GPUs (4 for MMLU to reduce per-GPU memory)
    if [ "$IS_MMLU" -eq 1 ]; then
        # MMLU: Use 4 GPUs, batch_size=1 to minimize per-GPU memory
        # Effective batch = 1 × 4 × 4 = 16
        BATCH_SIZE=1
        GRAD_ACCUM=4
    else
        # Standard: Effective batch = 4 × 2 × 2 = 16
        BATCH_SIZE=4
        GRAD_ACCUM=2
    fi
    LR_OVERRIDE=1e-4
    DEEPSPEED_FLAG="--fsdp"
    EFFECTIVE_BATCH=$((BATCH_SIZE * TOTAL_ACTUAL_GPUS * GRAD_ACCUM))
    echo "[$(hostname)] LoRA config: batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM, GPUs=$TOTAL_ACTUAL_GPUS, effective_batch=$EFFECTIVE_BATCH, save_steps=$SAVE_STEPS"
else
    # Full fine-tuning: 8 GPUs
    BATCH_SIZE=1
    DESIRED_EFFECTIVE_BATCH=16
    # Keep effective batch approximately stable even when total GPUs differ.
    GRAD_ACCUM=$(( (DESIRED_EFFECTIVE_BATCH + TOTAL_ACTUAL_GPUS - 1) / TOTAL_ACTUAL_GPUS ))
    if [ "$GRAD_ACCUM" -lt 1 ]; then
        GRAD_ACCUM=1
    fi
    LR_OVERRIDE=1e-5
    # Use DeepSpeed ZeRO-3 with CPU offloading for full fine-tuning
    DEEPSPEED_FLAG="--deep-speed"
    EFFECTIVE_BATCH=$((BATCH_SIZE * TOTAL_ACTUAL_GPUS * GRAD_ACCUM))
    echo "[$(hostname)] Full FT config: batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM, GPUs=$TOTAL_ACTUAL_GPUS, effective_batch=$EFFECTIVE_BATCH, save_steps=$SAVE_STEPS"
fi

# Extended timeout for multi-node synchronization
export TORCH_DISTRIBUTED_TIMEOUT=7200

# Start a background watchdog to detect stuck training
TRAINING_LOG="$CHECKPOINT_PATH/training.log"
HEARTBEAT_FILE="$CHECKPOINT_PATH/training_heartbeat.txt"
mkdir -p "$CHECKPOINT_PATH"

# Watchdog that monitors training progress
start_watchdog() {
    (
        while true; do
            sleep 300  # Check every 5 minutes
            if [ -f "$TRAINING_LOG" ]; then
                LAST_MOD=$(stat -c %Y "$TRAINING_LOG" 2>/dev/null || echo 0)
                NOW=$(date +%s)
                DIFF=$((NOW - LAST_MOD))
                
                # If log hasn't been updated in 1 hour, consider it stuck
                if [ $DIFF -gt 3600 ]; then
                    echo "[$(date)] WATCHDOG: Training appears stuck (log not updated for ${DIFF}s)" >> "$HEARTBEAT_FILE"
                else
                    echo "[$(date)] WATCHDOG: Training active (last update ${DIFF}s ago)" >> "$HEARTBEAT_FILE"
                fi
            fi
        done
    ) &
    WATCHDOG_PID=$!
    echo "[$(hostname)] Started watchdog (PID: $WATCHDOG_PID)"
}

# Retry logic for transient failures
MAX_RETRIES=3
RETRY_COUNT=0

start_watchdog

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "[$(hostname)] Training attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    
    # Run training with increased resilience
    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --nproc_per_node=$NUM_GPUS_PER_NODE \
      --node_rank=$SLURM_NODEID \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_conf timeout=900 \
      --max_restarts=3 \
      --start_method=spawn \
      /home/mmahaut/projects/paramem/paramem/fine_tune_with_ckpts.py \
      --model-name "$MODEL_NAME" \
      --dataset-name "$DATASET_NAME" \
      --output-dir "$CHECKPOINT_PATH" \
      --learning-rate=$LR_OVERRIDE \
      --num-train-epochs 5 \
      --per-device-train-batch-size $BATCH_SIZE \
      --gradient-accumulation-steps $GRAD_ACCUM \
      --save-steps $SAVE_STEPS \
      --logging-steps 10 \
      $DEEPSPEED_FLAG \
      $( [ "$USE_LORA" -eq 1 ] && echo '--use-lora --lora-r 8 --lora-alpha 32 --lora-dropout 0.1' )
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(hostname)] Training completed successfully"
        kill $WATCHDOG_PID 2>/dev/null || true
        exit 0
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "[$(hostname)] Training failed with exit code $EXIT_CODE, retrying in 60s..."
        sleep 60
        
        # Reset GPU state between retries
        nvidia-smi --gpu-reset 2>/dev/null || true
        
        # Generate new master port to avoid port conflicts
        export MASTER_PORT=$((29500 + RANDOM % 1000))
        echo "[$(hostname)] New MASTER_PORT: $MASTER_PORT"
    else
        echo "[$(hostname)] All retry attempts exhausted, failing"
        kill $WATCHDOG_PID 2>/dev/null || true
        exit $EXIT_CODE
    fi
done
