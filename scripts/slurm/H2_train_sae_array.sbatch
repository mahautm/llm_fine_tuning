#!/bin/bash
#SBATCH --job-name=H2_train_sae
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_array%A_task%a.out
#SBATCH --error=slurm_logs/%x_array%A_task%a.err

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

# Map SLURM array task ID to SAE architectures for hypothesis H2 (Architecture Independence)
TYPES=("standard" "topk" "batch_topk")
SAE_TYPE=${TYPES[$SLURM_ARRAY_TASK_ID]}

# Rich domain follow-up: train on OLMo-1B Wikidata activations.
ACTS_PATH="data/activations/allenai_OLMo-1B-hf_wikidata_Met7.csv_acts.pt"

echo "--- Training $SAE_TYPE SAE ---"
echo "Activations: $ACTS_PATH"

"$PYTHON_BIN" scripts/train_sae.py \
    --acts_path $ACTS_PATH \
    --type $SAE_TYPE \
    --out_dir results/sae_models \
    --epochs 5
