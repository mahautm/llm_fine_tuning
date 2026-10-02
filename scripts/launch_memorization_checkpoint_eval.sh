#!/bin/bash
# Checkpoint evaluation script for models4: memorization + performance + ID
# Called by fine_tune_with_ckpts.py after each checkpoint save

EXCLUDE_NODES="node044,node041"

# Environment variables expected from caller:
# - CHECKPOINT_PATH: path to the checkpoint directory
# - USE_LORA: 0 or 1

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH not set"
    exit 1
fi

if [ -z "$USE_LORA" ]; then
    echo "ERROR: USE_LORA not set"
    exit 1
fi

MEM_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/memorization_evaluation.py"
PERF_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/performance_evaluation.py"
ID_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/ID_matrixent_evaluation.py"
OUT_DIR="${CHECKPOINT_PATH}/slurm_logs"
mkdir -p "$OUT_DIR"

MEM_OUT="${OUT_DIR}/memorization_evaluation.out"
MEM_ERR="${OUT_DIR}/memorization_evaluation.err"
PERF_OUT="${OUT_DIR}/performance_evaluation.out"
PERF_ERR="${OUT_DIR}/performance_evaluation.err"
ID_OUT="${OUT_DIR}/ID_matrixent.out"
ID_ERR="${OUT_DIR}/ID_matrixent.err"

echo "=========================================="
echo "Launching Checkpoint Evaluation Suite"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "USE_LORA: $USE_LORA"
echo "=========================================="

# Get MODEL_NAME from environment or use default
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}

# Launch memorization evaluation job
MEM_JOBID=$(sbatch \
    --job-name=mem_eval \
    --gres=gpu:1 \
    --mem=100G \
    --partition=alien \
    --qos=alien \
    --time=02:00:00 \
    --exclude=$EXCLUDE_NODES \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
    --output="$MEM_OUT" \
    --error="$MEM_ERR" \
    --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $MEM_SCRIPT" | awk '{print $4}')

echo "Memorization evaluation job submitted: $MEM_JOBID"

# Launch performance evaluation after memorization to avoid GPU oversubscription.
PERF_JOBID=$(sbatch \
    --job-name=perf_eval \
    --gres=gpu:1 \
    --mem=120G \
    --partition=alien \
    --qos=alien \
    --time=02:00:00 \
    --exclude=$EXCLUDE_NODES \
    --dependency=afterok:$MEM_JOBID \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
    --output="$PERF_OUT" \
    --error="$PERF_ERR" \
    --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $PERF_SCRIPT" | awk '{print $4}')

echo "Performance evaluation job submitted: $PERF_JOBID"

# Launch ID evaluation after performance.
ID_JOBID=$(sbatch \
    --job-name=id_eval \
    --gres=gpu:1 \
    --mem=120G \
    --partition=alien \
    --qos=alien \
    --time=02:00:00 \
    --exclude=$EXCLUDE_NODES \
    --dependency=afterok:$PERF_JOBID \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
    --output="$ID_OUT" \
    --error="$ID_ERR" \
    --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $ID_SCRIPT" | awk '{print $4}')

echo "ID evaluation job submitted: $ID_JOBID"

# Wait for suite to complete.
echo "Waiting for evaluation suite to complete..."
while true; do
    STATE=$(sacct -j "$ID_JOBID" --format=State -n -P 2>/dev/null | head -n 1 | tr -d ' ')
    if [ "$STATE" = "COMPLETED" ]; then
        echo "Evaluation suite completed successfully."
        break
    fi

    if [[ "$STATE" =~ FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL ]]; then
        echo "ERROR: Evaluation suite failed with state: $STATE"
        echo "Check logs:"
        echo "  $MEM_ERR"
        echo "  $PERF_ERR"
        echo "  $ID_ERR"
        exit 1
    fi

    sleep 30
done

# Cleanup: delete checkpoint payload only after all required outputs are present.
if [ "${KEEP_CHECKPOINT:-0}" = "0" ]; then
    METRICS_FILE="${OUT_DIR}/memorization_metrics.json"

    if [ -s "$METRICS_FILE" ] && [ -s "$PERF_OUT" ] && [ -s "$ID_OUT" ]; then
        echo "Cleaning up checkpoint directory (keeping slurm_logs)..."
        find "$CHECKPOINT_PATH" -mindepth 1 -maxdepth 1 -not -name 'slurm_logs' -exec rm -rf {} +
        echo "Checkpoint cleanup completed."
    else
        echo "Skipping cleanup: one or more required evaluation outputs are missing/empty."
    fi
else
    echo "KEEP_CHECKPOINT is set. Preserving checkpoint directory."
fi

echo "=========================================="
echo "Checkpoint evaluation completed."
echo "Results saved under: ${OUT_DIR}"
echo "=========================================="
