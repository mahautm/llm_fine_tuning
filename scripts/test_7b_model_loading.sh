#!/bin/bash
#SBATCH --job-name=test_7b_model
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=0:30:00
#SBATCH --output=slurm_logs/test_7b_model_%j.out
#SBATCH --error=slurm_logs/test_7b_model_%j.err

# Quick test to verify 7B model can be loaded and fits in memory

echo "=========================================="
echo "Testing 7B Model Loading"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Activate environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate paramem

# Create log directory
mkdir -p /home/mmahaut/projects/paramem/slurm_logs

# Display GPU info
nvidia-smi

echo ""
echo "Testing Llama-3.1-8B model loading..."
echo ""

# Run test
srun --ntasks=1 python /home/mmahaut/projects/paramem/scripts/tests/test_model_loading.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32

echo ""
echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="
