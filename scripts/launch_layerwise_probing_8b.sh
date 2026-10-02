#!/bin/bash
# Launch layerwise probing jobs for 8B LoRA checkpoints at specific steps

# Key checkpoints to probe (not all to save compute)
CHECKPOINTS=(100 500 1000 1540)

# Benchmarks to probe
BENCHMARKS="mmlu,arc"

# Output base directory
OUTPUT_BASE="/home/mmahaut/projects/paramem/output/layerwise_probing_8b"

echo "Launching layerwise probing jobs for 8B LoRA checkpoints..."
echo "Benchmarks: $BENCHMARKS"
echo "Checkpoints: ${CHECKPOINTS[@]}"
echo ""

# Function to launch probing job for a specific dataset
launch_probing_dataset() {
    local dataset=$1
    local output_dir="${OUTPUT_BASE}/${dataset}"
    local log_dir="${OUTPUT_BASE}/logs/${dataset}"
    mkdir -p "$log_dir"
    
    local job_name="probe_8b_${dataset}"
    
    echo "Submitting: $job_name"
    
    # The script will automatically find the latest checkpoint for the dataset
    sbatch --job-name="$job_name" \
           --partition=alien \
           --qos=alien \
           --gres=gpu:2 \
           --mem=150G \
           --time=8:00:00 \
           --exclude=node044,node043 \
           --output="${log_dir}/probing.out" \
           --error="${log_dir}/probing.err" \
           --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd /home/mmahaut/projects/paramem && python paramem/layerwise_probing.py --dataset='$dataset' --benchmarks='$BENCHMARKS' --output-dir='$output_dir' --max-samples=500 --batch-size=4"
}

# Launch for pile dataset
echo "=== Pile Dataset ==="
launch_probing_dataset "pile"
sleep 1

# Launch for wikiplus dataset
echo "=== Wikiplus Dataset ==="
launch_probing_dataset "wikiplus"

echo ""
echo "✅ Layerwise probing jobs submitted!"
echo "📁 Results will be saved to: $OUTPUT_BASE"
echo ""
echo "Note: The script will automatically use the latest checkpoint for each dataset."
echo "To monitor progress: tail -f ${OUTPUT_BASE}/logs/*/probing.out"
