#!/bin/bash
# run_memorization_on_checkpoints.sh
# Batch script to run memorization evaluation on all existing checkpoints

EXCLUDE_NODES="node044,node041"
MEM_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/memorization_evaluation.py"

# Model directories to process
MODEL_DIRS=(
    "/home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-0-mmlu-lr1e-4"
    "/home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-0-pile-lr1e-4"
    "/home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4"
    "/home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4"
    "/home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4"
    "/home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4"
)

echo "========================================"
echo "Memorization Evaluation Batch Script"
echo "========================================"
echo ""

total_checkpoints=0
submitted_jobs=0

# Loop through each model directory
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠ Skipping non-existent directory: $MODEL_DIR"
        continue
    fi
    
    # Determine if this is LoRA or Full-FT
    if [[ "$MODEL_DIR" == *"fsdp-1"* ]]; then
        USE_LORA=1
        MODEL_TYPE="LoRA"
    else
        USE_LORA=0
        MODEL_TYPE="Full-FT"
    fi
    
    # Extract dataset name
    if [[ "$MODEL_DIR" == *"mmlu"* ]]; then
        DATASET="MMLU"
    elif [[ "$MODEL_DIR" == *"pile"* ]]; then
        DATASET="Pile"
    elif [[ "$MODEL_DIR" == *"wikiplus"* ]]; then
        DATASET="Wikiplus"
    else
        DATASET="Unknown"
    fi
    
    echo "Processing: $(basename $MODEL_DIR)"
    echo "  Type: $MODEL_TYPE, Dataset: $DATASET"
    
    # Find all checkpoint directories with slurm_logs
    checkpoint_count=0
    for CHECKPOINT_PATH in "$MODEL_DIR"/checkpoint-*/; do
        if [ -d "$CHECKPOINT_PATH/slurm_logs" ]; then
            CHECKPOINT_NUM=$(basename "$CHECKPOINT_PATH" | cut -d'-' -f2)
            
            # Check if memorization_metrics.out already exists
            MEM_OUT="${CHECKPOINT_PATH}slurm_logs/memorization_metrics.out"
            MEM_ERR="${CHECKPOINT_PATH}slurm_logs/memorization_metrics.err"
            
            if [ -f "$MEM_OUT" ]; then
                # Check if it's completed
                if grep -q "COMPLETED" "$MEM_OUT" 2>/dev/null; then
                    echo "    ✓ Checkpoint-$CHECKPOINT_NUM: Already completed"
                    checkpoint_count=$((checkpoint_count + 1))
                    continue
                else
                    echo "    ⟳ Checkpoint-$CHECKPOINT_NUM: Re-running (incomplete)"
                fi
            else
                echo "    → Checkpoint-$CHECKPOINT_NUM: Submitting job"
            fi
            
            # Submit memorization evaluation job
            JOB_NAME="mem_$(basename $MODEL_DIR | cut -d'-' -f1-3)_ckpt${CHECKPOINT_NUM}"
            sbatch --job-name="$JOB_NAME" \
                --gres=gpu:1 \
                --mem=100G \
                --partition=alien \
                --qos=alien \
                --time=02:00:00 \
                --exclude=$EXCLUDE_NODES \
                --export=CHECKPOINT_PATH="${CHECKPOINT_PATH%/}",USE_LORA=$USE_LORA \
                --output="$MEM_OUT" \
                --error="$MEM_ERR" \
                --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && srun --ntasks=1 poetry run python $MEM_SCRIPT"
            
            checkpoint_count=$((checkpoint_count + 1))
            submitted_jobs=$((submitted_jobs + 1))
            
            # Small delay to avoid overwhelming the scheduler
            sleep 2
        fi
    done
    
    total_checkpoints=$((total_checkpoints + checkpoint_count))
    echo "  Found $checkpoint_count checkpoints with slurm_logs"
    echo ""
done

echo "========================================"
echo "Summary:"
echo "  Total checkpoints processed: $total_checkpoints"
echo "  Jobs submitted: $submitted_jobs"
echo "========================================"
echo ""
echo "Monitor jobs with: squeue -u $USER | grep mem_"
echo "Check progress: tail -f models3/*/checkpoint-*/slurm_logs/memorization_metrics.out"
