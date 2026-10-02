#!/bin/bash
#SBATCH --job-name=H1_extract_rich_domains
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source ~/.bashrc
set -euo pipefail

PYTHON_BIN="/home/mmahaut/.conda/envs/paramem/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate paramem
        PYTHON_BIN="$(which python)"
    else
        echo "Cannot find paramem python env: $PYTHON_BIN"
        exit 1
    fi
fi

echo "--- Extracting richer domain activations (OLMo-1B) ---"
srun "$PYTHON_BIN" scripts/data/extract_rich_domain_activations.py --out_dir data/activations --batches 20
