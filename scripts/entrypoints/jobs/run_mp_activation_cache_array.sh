#!/bin/bash
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=node044,node034
#SBATCH --job-name=mp-cache-array
#SBATCH --array=1-8
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%A_%a_mp_cache.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%A_%a_mp_cache.err

source ~/.bashrc
set -euo pipefail
module load CUDA/12.1.0 || true

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

TASK_FILE="scripts/entrypoints/jobs/mp_activation_array_tasks.tsv"
TASK_LINE=$(grep -v '^#' "$TASK_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [[ -z "$TASK_LINE" ]]; then
  echo "No task line found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

IFS=$'\t' read -r MODEL_TAG MODEL_NAME DATASET_NAME DATA_PATH BATCH_SIZE LAYERS <<< "$TASK_LINE"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing data file: $DATA_PATH"
  exit 1
fi

ACT_DIR="results/mp_reservoir/activations/${MODEL_TAG}"
mkdir -p "$ACT_DIR"
OUT_PREFIX="$ACT_DIR/${DATASET_NAME}_layers_${LAYERS//,/}.act"

echo "MODEL_TAG=$MODEL_TAG"
echo "MODEL_NAME=$MODEL_NAME"
echo "DATASET_NAME=$DATASET_NAME"
echo "DATA_PATH=$DATA_PATH"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LAYERS=$LAYERS"

echo "Caching activations..."
srun "$PYTHON_BIN" /home/mmahaut/projects/paramem/paramem/extract_hidden.py \
  "$MODEL_NAME" \
  "$BATCH_SIZE" \
  "$DATA_PATH" \
  --input-key="query" \
  --out-pickle-prefix="$OUT_PREFIX"

echo "Done: ${OUT_PREFIX}.pickle"
