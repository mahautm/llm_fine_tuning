#!/bin/bash
#SBATCH --job-name=gap_experiments
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/H7_gap_experiments_%j.out
#SBATCH --error=slurm_logs/H7_gap_experiments_%j.err

source ~/.bashrc
conda activate paramem

echo "--- Running NeurIPS Gap Implementations ---"

# Step 1: LLM-as-a-judge Noise Probe and Architecture Yield
echo "1. Running Semantic Validity & Yield Probes..."
python scripts/eval_gaps.py \
    --local_api_base "http://10.80.63.42:8000/v1" \
    --local_model_id "meta-llama/Meta-Llama-3-8B-Instruct"

echo "Gap implementations completed!"