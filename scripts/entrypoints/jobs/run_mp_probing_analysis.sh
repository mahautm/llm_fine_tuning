#!/bin/bash
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --exclude=node044,node034
#SBATCH --job-name=mp-probing-analysis
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_probing.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_probing.err

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

srun "$PYTHON_BIN" scripts/analysis/run_mp_probing_analysis.py \
  --activations-root results/mp_reservoir/activations \
  --spectra-root results/mp_reservoir/spectra \
  --output-root results/mp_reservoir/analysis \
  --tag-suffix todo_exp1 \
  --k-neighbors 10 \
  --max-samples 1200
