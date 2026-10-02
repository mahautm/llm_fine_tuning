#!/bin/bash
#SBATCH --job-name=H3_eval_pareto_mp
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=02:00:00
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

ACTS_PATH_DOM="data/activations/allenai_OLMo-1B-hf_wikidata_Met7.csv_acts.pt"
ACTS_PATH_GEN="data/activations/allenai_OLMo-1B-hf_pile_Met7.txt_acts.pt"

echo "--- Phase 1: Evaluating Pareto Front (MSE vs Sparsity) ---"
"$PYTHON_BIN" scripts/evaluate_pareto.py \
    --models_dir results/sae_models \
    --test_acts $ACTS_PATH_DOM \
    --out_csv results/H3_pareto_front.csv

echo "--- Phase 2: Evaluating MP Threshold (Superposition Substrate) ---"
# Select standard model as baseline mapping for MP bulk edge
SAE_MODEL="results/sae_models/standard_sae_allenai_OLMo-1B-hf_wikidata_Met7.csv_ef8.pt"
F_OUT_DOM="results/sae_activations/f_acts_olmo_dom.pt"
F_OUT_GEN="results/sae_activations/f_acts_olmo_gen.pt"

"$PYTHON_BIN" scripts/evaluate_activations.py \
    --sae_model $SAE_MODEL \
    --acts_in $ACTS_PATH_DOM \
    --acts_out $F_OUT_DOM

"$PYTHON_BIN" scripts/evaluate_activations.py \
    --sae_model $SAE_MODEL \
    --acts_in $ACTS_PATH_GEN \
    --acts_out $F_OUT_GEN

# Map features (Gen vs Dom)
"$PYTHON_BIN" scripts/apply_mp_threshold.py \
    --f_gen $F_OUT_GEN \
    --f_dom $F_OUT_DOM \
    --out_json results/H3_mp_threshold_report.json

# Calibrated pass to reduce all-universal/all-dead collapse in difficult regimes.
"$PYTHON_BIN" scripts/apply_mp_threshold.py \
    --f_gen $F_OUT_GEN \
    --f_dom $F_OUT_DOM \
    --scale_mode mad \
    --clip_quantile 0.995 \
    --sigma_estimator quantile \
    --sigma_quantile 0.3 \
    --out_json results/H3_mp_threshold_report_calibrated.json
