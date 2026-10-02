#!/bin/bash
# Monitor models4 training and memorization evaluation jobs

echo "=========================================="
echo "Models4 Training & Evaluation Status"
echo "=========================================="
echo ""

# Show training jobs
echo "Training Jobs:"
squeue --me -o "%.10i %.20j %.8T %.10M %.9l %.6D %.20R" | grep "models4"

echo ""
echo "Memorization Evaluation Jobs:"
squeue --me -o "%.10i %.20j %.8T %.10M %.9l %.6D %.20R" | grep "mem_eval"

echo ""
echo "=========================================="
echo "Models4 Directory Contents:"
echo "=========================================="
echo ""

# Check if models4 exists
if [ -d "/home/mmahaut/projects/paramem/models4" ]; then
    for MODEL_DIR in /home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-*; do
        if [ -d "$MODEL_DIR" ]; then
            MODEL_NAME=$(basename "$MODEL_DIR")
            echo "Model: $MODEL_NAME"
            
            # Count checkpoints
            CHECKPOINT_COUNT=$(find "$MODEL_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l)
            echo "  Checkpoints: $CHECKPOINT_COUNT"
            
            # List checkpoints with memorization results
            echo "  Checkpoints with memorization results:"
            for CKPT in $(find "$MODEL_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V); do
                CKPT_NAME=$(basename "$CKPT")
                if [ -f "$CKPT/slurm_logs/memorization_results.json" ]; then
                    echo "    ✓ $CKPT_NAME (results available)"
                elif [ -d "$CKPT/slurm_logs" ]; then
                    echo "    ⏳ $CKPT_NAME (evaluation in progress)"
                else
                    echo "    ⋯ $CKPT_NAME (pending evaluation)"
                fi
            done
            echo ""
        fi
    done
else
    echo "models4 directory does not exist yet."
fi

echo ""
echo "=========================================="
echo "To launch training:"
echo "  bash /home/mmahaut/projects/paramem/scripts/launch_models4_wikiplus_training.sh"
echo ""
echo "To watch continuously:"
echo "  watch -n 30 bash /home/mmahaut/projects/paramem/scripts/monitor_models4.sh"
echo "=========================================="
