#!/bin/bash
#SBATCH --job-name=H1_extract_acts
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source ~/.bashrc
conda activate paramem

echo "--- Extrating Activations from host models ---"
# Using uv format as requested
python scripts/data/extract_all_activations.py --out_dir data/activations --batches 20
