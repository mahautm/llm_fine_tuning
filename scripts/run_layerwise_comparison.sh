#!/bin/bash
#SBATCH --job-name=layerwise_comparison
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=4:00:00
#SBATCH --exclude=node044
#SBATCH --output=slurm_logs/layerwise_comparison_%j.out
#SBATCH --error=slurm_logs/layerwise_comparison_%j.err

# Layerwise Representation Comparison Experiment
# Compares Full FT, LoRA FT, and Original models using II and Neighbourhood Overlap

echo "=========================================="
echo "Layerwise Representation Comparison"
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
mkdir -p /home/mmahaut/projects/paramem/output/layerwise_comparison
mkdir -p /home/mmahaut/projects/paramem/slurm_logs

# Run experiment for pile dataset
echo ""
echo "Processing PILE dataset..."
python /home/mmahaut/projects/paramem/paramem/layerwise_representation_comparison.py \
    --models-dir /home/mmahaut/projects/paramem/models2 \
    --benchmark-file /home/mmahaut/projects/paramem/benchmark/dev.jsonl \
    --output-dir /home/mmahaut/projects/paramem/output/layerwise_comparison/pile \
    --dataset pile \
    --limit 1000 \
    --k-neighbors 50 \
    --batch-size 8

echo ""
echo "Processing WIKIPLUS dataset..."
python /home/mmahaut/projects/paramem/paramem/layerwise_representation_comparison.py \
    --models-dir /home/mmahaut/projects/paramem/models2 \
    --benchmark-file /home/mmahaut/projects/paramem/benchmark/dev.jsonl \
    --output-dir /home/mmahaut/projects/paramem/output/layerwise_comparison/wikiplus \
    --dataset wikiplus \
    --limit 1000 \
    --k-neighbors 50 \
    --batch-size 8

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
