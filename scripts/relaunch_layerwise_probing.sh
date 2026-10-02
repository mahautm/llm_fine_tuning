#!/bin/bash
# Relaunch only layerwise probing for existing checkpoints
# Usage: bash scripts/relaunch_layerwise_probing.sh [checkpoint_path]
# Or to process all checkpoints: bash scripts/relaunch_layerwise_probing.sh --all

EXCLUDE_NODES="node044"
PROBING_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/layerwise_probing_checkpoint.py"

# Function to launch probing for a single checkpoint
launch_probing() {
    local CHECKPOINT_PATH=$1
    local USE_LORA=$2
    
    local OUT_DIR="${CHECKPOINT_PATH}/slurm_logs"
    mkdir -p "$OUT_DIR"
    
    local PROBING_OUT="${OUT_DIR}/layerwise_probing.out"
    local PROBING_ERR="${OUT_DIR}/layerwise_probing.err"
    
    echo "Launching layerwise probing for: $CHECKPOINT_PATH"
    
    PROBING_JOBID=$(sbatch --job-name=layerwise_probing --gres=gpu:2 --mem=200G --partition=alien --qos=alien \
        --time=02:00:00 \
        --exclude=$EXCLUDE_NODES \
        --output="$PROBING_OUT" --error="$PROBING_ERR" \
        --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA \
        --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd paramem && poetry run python $PROBING_SCRIPT" | awk '{print $4}')
    
    echo "  Job ID: $PROBING_JOBID"
}

# Function to detect if checkpoint uses LoRA
detect_lora() {
    local checkpoint_path=$1
    if [[ "$checkpoint_path" == *"-fsdp-1-"* ]]; then
        echo 1
    else
        echo 0
    fi
}

# Main logic
if [ "$1" = "--all" ]; then
    echo "=== Relaunching layerwise probing for all checkpoints ==="
    
    # Find all checkpoint directories that have checkpoint-* subdirectories
    for model_dir in /home/mmahaut/projects/paramem/models3/Llama-3.1-8B-Instruct-fsdp-*; do
        if [ -d "$model_dir" ]; then
            echo ""
            echo "Processing model: $(basename $model_dir)"
            
            USE_LORA=$(detect_lora "$model_dir")
            
            # Find checkpoint subdirectories
            for ckpt_dir in "$model_dir"/checkpoint-*; do
                if [ -d "$ckpt_dir" ]; then
                    # Check if probing already completed
                    PROBING_OUT="${ckpt_dir}/slurm_logs/layerwise_probing.out"
                    if [ -f "$PROBING_OUT" ] && grep -q "COMPLETED" "$PROBING_OUT"; then
                        echo "  Skipping $(basename $ckpt_dir) - already completed"
                    else
                        launch_probing "$ckpt_dir" "$USE_LORA"
                        sleep 5
                    fi
                fi
            done
        fi
    done
    
elif [ -n "$1" ]; then
    # Single checkpoint provided
    CHECKPOINT_PATH=$1
    
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
        exit 1
    fi
    
    USE_LORA=$(detect_lora "$CHECKPOINT_PATH")
    launch_probing "$CHECKPOINT_PATH" "$USE_LORA"
    
else
    echo "Usage:"
    echo "  Relaunch for specific checkpoint:"
    echo "    bash scripts/relaunch_layerwise_probing.sh /path/to/checkpoint"
    echo ""
    echo "  Relaunch for all incomplete checkpoints:"
    echo "    bash scripts/relaunch_layerwise_probing.sh --all"
    exit 1
fi

echo ""
echo "=== Layerwise probing jobs launched ==="
