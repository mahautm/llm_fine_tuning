# Exclude node044 from Slurm jobs
EXCLUDE_NODES="node044"

# when this job is launched by paremem.fine_tune_with_ckpts.py we are given "CHECKPOINT_DIR": checkpoint_dir and "USE_LORA": int(args.use_lora)
# Optional: KEEP_CHECKPOINT=1 to preserve checkpoint after evaluation
# Set variables
PERF_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/performance_evaluation.py"
ID_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/ID_matrixent_evaluation.py"
PROBING_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/layerwise_probing_checkpoint.py"
MEM_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/memorization_evaluation.py"
MP_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/mp_checkpoint_evaluation.py"
OUT_DIR="${CHECKPOINT_PATH}/slurm_logs"
mkdir -p "$OUT_DIR"

LOCK_DIR="${OUT_DIR}/.eval_lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Evaluation lock exists for $CHECKPOINT_PATH, skipping duplicate launch."
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

# Toggle which evaluations to run.
# Defaults are space/queue-friendly: run memorization + MP only.
# Set RUN_PERF=1 and/or RUN_ID=1 when you explicitly need them.
RUN_PERF="${RUN_PERF:-0}"
RUN_ID="${RUN_ID:-0}"
RUN_PROBING="${RUN_PROBING:-0}"
RUN_MEM="${RUN_MEM:-1}"
RUN_MP="${RUN_MP:-1}"
MODEL_NAME="${MODEL_NAME:-}"

# Preserve every Nth checkpoint even after successful evaluation (0 disables).
KEEP_EVERY_N="${KEEP_EVERY_N:-0}"

PERF_OUT="${OUT_DIR}/performance_evaluation.out"
PERF_ERR="${OUT_DIR}/performance_evaluation.err"
ID_OUT="${OUT_DIR}/ID_matrixent.out"
ID_ERR="${OUT_DIR}/ID_matrixent.err"
PROBING_OUT="${OUT_DIR}/layerwise_probing.out"
PROBING_ERR="${OUT_DIR}/layerwise_probing.err"
MEM_OUT="${OUT_DIR}/memorization_metrics.out"
MEM_ERR="${OUT_DIR}/memorization_metrics.err"
MP_OUT="${OUT_DIR}/mp_checkpoint_evaluation.out"
MP_ERR="${OUT_DIR}/mp_checkpoint_evaluation.err"

if [ -f "${OUT_DIR}/memorization_metrics.json" ] && [ -f "${OUT_DIR}/mp_checkpoint_metrics.json" ]; then
    echo "Required evaluation artifacts already exist for $CHECKPOINT_PATH; skipping relaunch."
    exit 0
fi

if [ "$RUN_PERF" = "1" ]; then
    PERF_JOBID=$(sbatch --job-name=perf_eval --gres=gpu:1 --mem=120G --partition=alien --qos=alien \
        --exclude=$EXCLUDE_NODES \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
        --output="$PERF_OUT" --error="$PERF_ERR" \
        --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $PERF_SCRIPT" | awk '{print $4}')
fi

if [ "$RUN_ID" = "1" ]; then
    ID_JOBID=$(sbatch --job-name=id_matrixent --gres=gpu:1 --mem=120G --partition=alien --qos=alien \
        --exclude=$EXCLUDE_NODES \
        --output="$ID_OUT" --error="$ID_ERR" \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
        --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $ID_SCRIPT" | awk '{print $4}')
fi

if [ "$RUN_PROBING" = "1" ]; then
    PROBING_JOBID=$(sbatch --job-name=layerwise_probing --gres=gpu:1 --mem=120G --partition=alien --qos=alien \
        --time=02:00:00 \
        --exclude=$EXCLUDE_NODES \
        --output="$PROBING_OUT" --error="$PROBING_ERR" \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
        --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $PROBING_SCRIPT" | awk '{print $4}')
fi

if [ "$RUN_MEM" = "1" ]; then
    MEM_JOBID=$(sbatch --job-name=mem_eval --gres=gpu:1 --mem=120G --partition=alien --qos=alien \
        --exclude=$EXCLUDE_NODES \
        --output="$MEM_OUT" --error="$MEM_ERR" \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
        --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $MEM_SCRIPT" | awk '{print $4}')
fi

if [ "$RUN_MP" = "1" ]; then
    MP_JOBID=$(sbatch --job-name=mp_eval --gres=gpu:1 --mem=120G --partition=alien --qos=alien \
        --exclude=$EXCLUDE_NODES \
        --output="$MP_OUT" --error="$MP_ERR" \
        --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA,MODEL_NAME=$MODEL_NAME \
        --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && poetry run python $MP_SCRIPT" | awk '{print $4}')
fi

# Wait for enabled jobs by SLURM job IDs. This avoids hanging when a job fails
# without printing a terminal "COMPLETED" marker.
JOB_IDS=()

if [ "$RUN_PERF" = "1" ] && [ -n "$PERF_JOBID" ]; then
    JOB_IDS+=("$PERF_JOBID")
fi
if [ "$RUN_ID" = "1" ] && [ -n "$ID_JOBID" ]; then
    JOB_IDS+=("$ID_JOBID")
fi
if [ "$RUN_PROBING" = "1" ] && [ -n "$PROBING_JOBID" ]; then
    JOB_IDS+=("$PROBING_JOBID")
fi
if [ "$RUN_MEM" = "1" ] && [ -n "$MEM_JOBID" ]; then
    JOB_IDS+=("$MEM_JOBID")
fi
if [ "$RUN_MP" = "1" ] && [ -n "$MP_JOBID" ]; then
    JOB_IDS+=("$MP_JOBID")
fi

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    while true; do
        ANY_ACTIVE=false
        for jid in "${JOB_IDS[@]}"; do
            if squeue -h -j "$jid" | grep -q .; then
                ANY_ACTIVE=true
                break
            fi
        done
        if [ "$ANY_ACTIVE" = "false" ]; then
            break
        fi
        sleep 30
    done
fi

# Validate required artifacts before cleanup.
CAN_CLEAN=true
if [ "$RUN_MEM" = "1" ] && [ ! -f "${OUT_DIR}/memorization_metrics.json" ]; then
    CAN_CLEAN=false
    echo "Missing memorization_metrics.json; will keep checkpoint."
fi
if [ "$RUN_MEM" = "1" ] && [ -f "${OUT_DIR}/memorization_metrics.json" ]; then
    if grep -Eq '^\s*\{\s*\}\s*$' "${OUT_DIR}/memorization_metrics.json"; then
        CAN_CLEAN=false
        echo "memorization_metrics.json is empty (likely model-load failure); will keep checkpoint."
    fi
fi
if [ "$RUN_MP" = "1" ] && [ ! -f "${OUT_DIR}/mp_checkpoint_metrics.json" ]; then
    CAN_CLEAN=false
    echo "Missing mp_checkpoint_metrics.json; will keep checkpoint."
fi

# Keep every Nth checkpoint for recovery/audits if requested.
CKPT_BASENAME=$(basename "$CHECKPOINT_PATH")
CKPT_NUM=${CKPT_BASENAME#checkpoint-}
if [ "$KEEP_EVERY_N" != "0" ] && [ -n "$CKPT_NUM" ] && [ "$CKPT_NUM" -gt 0 ] 2>/dev/null; then
    if [ $((CKPT_NUM % KEEP_EVERY_N)) -eq 0 ]; then
        CAN_CLEAN=false
        echo "KEEP_EVERY_N=$KEEP_EVERY_N -> preserving checkpoint-$CKPT_NUM."
    fi
fi

# Delete checkpoint payload unless KEEP_CHECKPOINT=1 or checks failed.
if [ "${KEEP_CHECKPOINT:-0}" = "0" ] && [ "$CAN_CLEAN" = "true" ]; then
    echo "Cleaning up checkpoint directory (keeping slurm_logs)..."
    find "$CHECKPOINT_PATH" -mindepth 1 -maxdepth 1 -not -name 'slurm_logs' -exec rm -rf {} +
    echo "Checkpoint cleanup completed."
else
    echo "Preserving checkpoint directory (KEEP_CHECKPOINT set or required artifacts missing)."
fi