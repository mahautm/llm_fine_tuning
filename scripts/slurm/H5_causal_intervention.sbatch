#!/bin/bash
#SBATCH --job-name=H5_causal_intervention
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source ~/.bashrc
set -euo pipefail
conda activate paramem

SAE_MODEL="results/sae_models/standard_sae_EleutherAI_pythia-70m_imdb_ef8.pt"
FEATURE_IDX=0 
STEER_COEFF=20.0

echo "--- Phase 5: Causal Intervention / Feature Steering ---"
python scripts/causal_intervention.py \
    --model "EleutherAI/pythia-70m" \
    --layer "gpt_neox.layers.3" \
    --sae $SAE_MODEL \
    --feature $FEATURE_IDX \
    --coeff $STEER_COEFF \
    --prompt "The movie I watched last night was generally"

echo "--- Phase 5: Causal Intervention / Sweep Evaluation ---"
python scripts/causal_intervention_sweep.py \
    --model "EleutherAI/pythia-70m" \
    --layer "gpt_neox.layers.3" \
    --sae $SAE_MODEL \
    --feature $FEATURE_IDX \
    --out results/H5_sweep_results.csv
