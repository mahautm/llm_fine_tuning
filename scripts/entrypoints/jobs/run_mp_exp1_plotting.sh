#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclude=node044,node034
#SBATCH --job-name=mp-exp1-plots
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_exp1_plots.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_exp1_plots.err

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

cd /home/mmahaut/projects/paramem

srun "$PYTHON_BIN" scripts/plots/plot_mp_exp1_heatmaps.py \
  --spectra-root results/mp_reservoir/spectra \
  --figures-root results/mp_reservoir/figures \
  --tag-suffix todo_exp1
