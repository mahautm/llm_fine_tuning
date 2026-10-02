#!/bin/bash
#SBATCH --job-name=layerwise_probing
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=8:00:00
#SBATCH --exclude=node044
#SBATCH --output=slurm_logs/layerwise_probing_%j.out
#SBATCH --error=slurm_logs/layerwise_probing_%j.err

# Layerwise Probing Experiment
# Tests MMLU and other benchmark performance using single-layer representations

echo "=========================================="
echo "Layerwise Probing Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Activate environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate paramem

# Ensure output directories exist
mkdir -p /home/mmahaut/projects/paramem/output/layerwise_probing
mkdir -p /home/mmahaut/projects/paramem/slurm_logs

# Define benchmarks (comma-separated) - will be loaded from Hugging Face
BENCHMARKS="mmlu,arc,hellaswag,winogrande,truthful_qa,openbookqa"

# Run experiment for pile dataset
echo ""
echo "Processing PILE dataset..."
python /home/mmahaut/projects/paramem/paramem/layerwise_probing.py \
    --models-dir /home/mmahaut/projects/paramem/models2 \
    --benchmark-data-dir /home/mmahaut/projects/paramem/benchmark/ \
    --output-dir /home/mmahaut/projects/paramem/output/layerwise_probing/pile \
    --dataset pile \
    --benchmarks "$BENCHMARKS" \
    --max-samples 1000 \
    --batch-size 8

echo ""
echo "Processing WIKIPLUS dataset..."
python /home/mmahaut/projects/paramem/paramem/layerwise_probing.py \
    --models-dir /home/mmahaut/projects/paramem/models2 \
    --benchmark-data-dir /home/mmahaut/projects/paramem/benchmark/ \
    --output-dir /home/mmahaut/projects/paramem/output/layerwise_probing/wikiplus \
    --dataset wikiplus \
    --benchmarks "$BENCHMARKS" \
    --max-samples 1000 \
    --batch-size 8

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
