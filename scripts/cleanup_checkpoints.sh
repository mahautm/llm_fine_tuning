#!/bin/bash
# Clean up MMLU checkpoints - keep only slurm_logs for half of them
# Keep the last checkpoint fully intact

MODEL_DIR="${1:-/home/mmahaut/projects/paramem/models3}"

for model_path in "$MODEL_DIR"/Llama-3.1-8B-Instruct-fsdp-*-mmlu-*; do
    if [ ! -d "$model_path" ]; then
        continue
    fi
    
    model_name=$(basename "$model_path")
    echo "========================================"
    echo "Processing: $model_name"
    echo "========================================"
    
    # Get all checkpoints sorted numerically
    checkpoints=($(ls -d "$model_path"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "  No checkpoints found"
        continue
    fi
    
    total=${#checkpoints[@]}
    last_idx=$((total - 1))
    
    echo "  Total checkpoints: $total"
    echo "  Last checkpoint: $(basename ${checkpoints[$last_idx]})"
    
    # Keep every other checkpoint (starting from index 1, 3, 5, ...), clean the rest
    # Always keep the last one fully
    for i in "${!checkpoints[@]}"; do
        ckpt="${checkpoints[$i]}"
        ckpt_name=$(basename "$ckpt")
        
        # Keep last checkpoint fully
        if [ $i -eq $last_idx ]; then
            echo "  ✓ Keeping full: $ckpt_name (last checkpoint)"
            continue
        fi
        
        # Clean all other checkpoints
        echo "  🧹 Cleaning: $ckpt_name (keeping slurm_logs)"
        
        # Check what exists before cleaning
        if [ ! -d "$ckpt/slurm_logs" ]; then
            echo "     ⚠️  No slurm_logs directory, skipping"
            continue
        fi
        
        # Create temporary directory to preserve files
        temp_dir=$(mktemp -d)
        
        # Copy slurm_logs to temp
        cp -r "$ckpt/slurm_logs" "$temp_dir/"
        
        # Remove all contents
        rm -rf "$ckpt"/*
        
        # Restore slurm_logs
        mv "$temp_dir/slurm_logs" "$ckpt/"
        
        # Cleanup temp
        rmdir "$temp_dir"
        
        echo "     ✅ Cleaned, slurm_logs preserved"
    done
    
    echo ""
done

echo "========================================"
echo "Checkpoint cleanup complete!"
echo "========================================"
