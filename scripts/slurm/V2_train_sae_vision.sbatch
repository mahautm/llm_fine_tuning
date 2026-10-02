#!/bin/bash
#SBATCH --job-name=V2_train_vision_sae
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

source ~/.bashrc
conda activate paramem

python scripts/train_sae.py \
    --acts_path data/activations/google_vit-base-patch16-224_cifar100_acts.pt \
    --expansion_factor 8 \
    --type standard \
    --out_dir results/sae_models
