#!/bin/bash
#SBATCH --job-name=H4_auto_annotation
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=01:00:00
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

REPORT="${REPORT:-results/H3_mp_threshold_report.json}"
F_ACTS="${F_ACTS:-results/sae_activations/f_acts_olmo_dom.pt}"
DATA="${DATA:-data/wikidata_Met7.csv}"
OUT="${OUT:-results/H4_llm_auto_annotations.json}"
LOCAL_API_BASE="${LOCAL_API_BASE:-http://10.80.63.42:8000/v1}"
LOCAL_MODEL_ID="${LOCAL_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"

echo "--- Phase 4: Illusion of Interpretability Auto-Annotation ---"
"$PYTHON_BIN" scripts/eval_auto_annotation.py \
    --report $REPORT \
    --f_acts $F_ACTS \
    --data $DATA \
    --out $OUT \
    --local_api_base "$LOCAL_API_BASE" \
    --local_model_id "$LOCAL_MODEL_ID"
