#!/bin/bash
#SBATCH --job-name=test_7b_lora
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --time=2:00:00
#SBATCH --exclude=node044
#SBATCH --output=slurm_logs/test_7b_lora_%j.out
#SBATCH --error=slurm_logs/test_7b_lora_%j.err

# Quick test: LoRA fine-tuning of 7B model with FSDP on 2 GPUs
# This runs for just a few steps to verify memory and configuration

echo "=========================================="
echo "Testing 7B LoRA Fine-Tuning with FSDP"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "=========================================="

# Activate environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate paramem

# Set CUDA_HOME for DeepSpeed (if not already set)
if [ -z "$CUDA_HOME" ]; then
    # Try to find CUDA installation
    if command -v nvcc &> /dev/null; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        echo "Set CUDA_HOME=$CUDA_HOME"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Set CUDA_HOME=$CUDA_HOME"
    else
        echo "Warning: Could not find CUDA installation. DeepSpeed may fail."
    fi
fi

# Create output directory
mkdir -p /home/mmahaut/projects/paramem/slurm_logs
mkdir -p /home/mmahaut/projects/paramem/models2/test_7b_lora

# Display GPU info
nvidia-smi

echo ""
echo "Configuration:"
echo "  Model: meta-llama/Llama-3.1-8B-Instruct"
echo "  GPUs: 2"
echo "  Training Mode: LoRA Fine-Tuning"
echo "  FSDP: Enabled"
echo "  LoRA rank: 8"
echo ""

cd /home/mmahaut/projects/paramem

# Run with torchrun for distributed setup
# Note: gradient-checkpointing and bf16 are hardcoded in fine_tune_with_ckpts.py
torchrun --nproc_per_node=2 --nnodes=1 \
  paramem/fine_tune_with_ckpts.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --dataset-name "/home/mmahaut/projects/paramem/data/pile_19_token_sample.txt" \
  --output-dir "/home/mmahaut/projects/paramem/models2/test_7b_lora" \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --num-train-epochs 1 \
  --save-steps 100 \
  --logging-steps 1 \
  --max-grad-norm 1.0 \
  --fsdp \
  --n-samples 100

echo ""
echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="
