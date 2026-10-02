#!/bin/bash
#SBATCH --job-name=V3_eval_mp_vision
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out

source ~/.bashrc
conda activate paramem

ACTS_PATH_GEN="data/activations/google_vit-base-patch16-224_cifar100_acts.pt"
ACTS_PATH_DOM="data/activations/google_vit-base-patch16-224_cifar10_acts.pt"
SAE_MODEL="results/sae_models/standard_sae_google_vit-base-patch16-224_cifar100_ef8.pt"

F_OUT_GEN="results/sae_activations/f_acts_vit_gen.pt"
F_OUT_DOM="results/sae_activations/f_acts_vit_dom.pt"

# Extract active variables from the SAE
python scripts/evaluate_activations.py --sae_model $SAE_MODEL --acts_in $ACTS_PATH_GEN --acts_out $F_OUT_GEN
python scripts/evaluate_activations.py --sae_model $SAE_MODEL --acts_in $ACTS_PATH_DOM --acts_out $F_OUT_DOM

# Apply MP Threshold and isolate structural visual categories 
python scripts/apply_mp_threshold.py --f_gen $F_OUT_GEN --f_dom $F_OUT_DOM --out_json results/V3_mp_vision_report.json
