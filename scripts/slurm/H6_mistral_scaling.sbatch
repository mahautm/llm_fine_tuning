#!/bin/bash
#SBATCH --job-name=Mistral_Scaling_SAE
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/H6_mistral_scaling_%j.out

source ~/.bashrc
conda activate paramem

echo "--- Launching Distributed SAE Extraction & Training on Mistral-7B ---"
# Actual massive dataset extraction (e.g. 10T tokens on multi-node) requires orchestrating distributed shards.
# This script serves as the scaled-up entrypoint targeting Mistral-7B's layer 16 (mid-point).

# 1. Extract Activations
echo "Extracting Mistral-7B-v0.3 activations..."
python scripts/data/extract_all_activations.py \
    --model mistralai/Mistral-7B-v0.3 \
    --layer model.layers.16 \
    --dataset data/pile_19_short.txt \
    --batches 100 \
    --out_dir data/mistral_acts

# 2. Train the Sparse Autoencoder
echo "Training Mistral-7B SAE..."
python scripts/train_sae.py \
    --acts_path data/mistral_acts/mistralai_Mistral-7B-v0.3_pile_19_short.txt_acts.pt \
    --expansion_factor 16 \
    --l1 0.00005 \
    --batch_size 1024 \
    --epochs 5 \
    --out_dir results/sae_models \
    --type standard

echo "Mistral-7B scaling execution completed!"
