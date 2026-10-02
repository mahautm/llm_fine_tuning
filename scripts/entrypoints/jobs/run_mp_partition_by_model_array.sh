#!/bin/bash
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --exclude=node044,node034
#SBATCH --job-name=mp-partition-array
#SBATCH --array=1-2
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%A_%a_mp_partition.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%A_%a_mp_partition.err

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

MODEL_TAGS=("pythia1b" "pythia14b")
LAYERS="0,4,8,12,16"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_TAG="${MODEL_TAGS[$TASK_INDEX]}"

ACT_DIR="results/mp_reservoir/activations/${MODEL_TAG}"
SPEC_DIR="results/mp_reservoir/spectra/${MODEL_TAG}"
mkdir -p "$SPEC_DIR"

PILE="${ACT_DIR}/pile19short_layers_0481216.act.pickle"
MMLU="${ACT_DIR}/mmlushort_layers_0481216.act.pickle"
MET7="${ACT_DIR}/pilemet7_layers_0481216.act.pickle"
MIS7="${ACT_DIR}/pilemis7_layers_0481216.act.pickle"

for f in "$PILE" "$MMLU" "$MET7" "$MIS7"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing activation cache: $f"
    exit 1
  fi
done

TAG="${MODEL_TAG}_todo_exp1"

echo "Running MP partition for model tag: $MODEL_TAG"
srun "$PYTHON_BIN" scripts/analysis/run_mp_partition_stability.py \
  --input "pile19short=$PILE" \
  --input "mmlushort=$MMLU" \
  --input "pilemet7=$MET7" \
  --input "pilemis7=$MIS7" \
  --layers "$LAYERS" \
  --output-dir "$SPEC_DIR" \
  --tag "$TAG"

echo "Done: $SPEC_DIR"
