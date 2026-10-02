#!/bin/bash
#SBATCH --job-name=V4_V5_vision_pipeline
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out

source ~/.bashrc
conda activate paramem

echo "--- V4 Vision Annotation ---"
python scripts/eval_vision_auto_annotation.py \
    --report results/V3_mp_vision_report.json \
    --f_acts results/sae_activations/f_acts_vit_dom.pt \
    --dataset cifar10 \
    --out results/V4_vision_annotations.json

echo "--- V5 Vision Intervention Sweep ---"
# Pick the most prominent visual domain feature
FEATURE_IDX=$(python -c "import json; print(json.load(open('results/V3_mp_vision_report.json')).get('Categories', {}).get('domain', [0])[0])")

python scripts/causal_intervention_vision.py \
    --model "google/vit-base-patch16-224" \
    --layer "encoder.layer.6" \
    --sae results/sae_models/standard_sae_google_vit-base-patch16-224_cifar100_ef8.pt \
    --feature $FEATURE_IDX \
    --dataset cifar10 \
    --out results/V5_vision_sweep.csv
