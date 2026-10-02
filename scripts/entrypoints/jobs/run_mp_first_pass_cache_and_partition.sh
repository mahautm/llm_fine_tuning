#!/bin/bash
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=node044
#SBATCH --job-name=mp-first-pass
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_first_pass.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_first_pass.err

set -euo pipefail

source ~/.bashrc
module load CUDA/12.1.0 || true
conda activate paramem

cd /home/mmahaut/projects/paramem

MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-70m}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LAYERS="${LAYERS:-0,2,4,6}"
TAG="${TAG:-pythia70m_pile_vs_mmlu_short}"

ACT_DIR="results/mp_reservoir/activations"
SPEC_DIR="results/mp_reservoir/spectra"
mkdir -p "$ACT_DIR" "$SPEC_DIR"

PILE_TXT="data/pile_19_short.txt"
MMLU_TXT="data/mmlu_train_short.txt"
PILE_OUT_PREFIX="$ACT_DIR/pile_19_short_${TAG}"
MMLU_OUT_PREFIX="$ACT_DIR/mmlu_train_short_${TAG}"

echo "[1/3] Caching activations for pile short"
srun python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \
  "$MODEL_NAME" \
  "$BATCH_SIZE" \
  "$PILE_TXT" \
  --out-pickle-prefix="$PILE_OUT_PREFIX"

echo "[2/3] Caching activations for mmlu short"
srun python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \
  "$MODEL_NAME" \
  "$BATCH_SIZE" \
  "$MMLU_TXT" \
  --out-pickle-prefix="$MMLU_OUT_PREFIX"

echo "[3/3] Running MP partition stability analysis"
srun python scripts/analysis/run_mp_partition_stability.py \
  --input "pile=$PILE_OUT_PREFIX.pickle" \
  --input "mmlu=$MMLU_OUT_PREFIX.pickle" \
  --layers "$LAYERS" \
  --output-dir "$SPEC_DIR" \
  --tag "$TAG"

echo "MP first pass complete. Outputs in $SPEC_DIR"
