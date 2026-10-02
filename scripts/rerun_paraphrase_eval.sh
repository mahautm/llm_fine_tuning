#!/bin/bash

# Script to rerun paraphrase evaluation with LLM-generated paraphrases
# Only runs on checkpoints that have full model weights (merged_single_model directory)

BASE_DIR="/home/mmahaut/projects/paramem"
MODELS_DIR="${BASE_DIR}/models3"

echo "==================================================================="
echo "Rerunning Paraphrase Evaluation with LLM-Generated Paraphrases"
echo "==================================================================="
echo ""

# Find all checkpoints with full model weights
declare -a CHECKPOINTS=()

for model_dir in ${MODELS_DIR}/Llama-3.1-8B-Instruct-fsdp-*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(basename "$model_dir")
    echo "Checking model: $model_name"
    
    for ckpt in "$model_dir"/checkpoint-*; do
        if [ ! -d "$ckpt" ]; then
            continue
        fi
        
        ckpt_name=$(basename "$ckpt")
        
        # Check if this checkpoint has full model weights (Full-FT) or LoRA adapter
        if [ -d "$ckpt/merged_single_model" ]; then
            CHECKPOINTS+=("$ckpt|fullft")
            echo "  ✓ $ckpt_name (Full-FT model)"
        elif [ -f "$ckpt/adapter_model.safetensors" ]; then
            CHECKPOINTS+=("$ckpt|lora")
            echo "  ✓ $ckpt_name (LoRA adapter)"
        else
            echo "  ✗ $ckpt_name (no model weights, skipping)"
        fi
    done
    echo ""
done

echo "==================================================================="
echo "Found ${#CHECKPOINTS[@]} checkpoints (Full-FT + LoRA)"
echo "==================================================================="
echo ""

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found with model weights. Exiting."
    exit 0
fi

# Launch evaluation jobs for each checkpoint
for checkpoint_entry in "${CHECKPOINTS[@]}"; do
    # Split entry into path and type
    IFS='|' read -r ckpt_path ckpt_type <<< "$checkpoint_entry"
    
    model_dir=$(dirname "$ckpt_path")
    model_name=$(basename "$model_dir")
    ckpt_name=$(basename "$ckpt_path")
    
    echo "Launching paraphrase evaluation for: $model_name / $ckpt_name ($ckpt_type)"
    
    # Create job script
    job_script="${BASE_DIR}/scripts/temp_paraphrase_eval_${model_name}_${ckpt_name}.sh"
    
    cat > "$job_script" << 'EOFSCRIPT'
#!/bin/bash
#SBATCH --job-name=paraphrase_eval
#SBATCH --output=__CHECKPOINT_PATH__/slurm_logs/paraphrase_eval_%j.out
#SBATCH --error=__CHECKPOINT_PATH__/slurm_logs/paraphrase_eval_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=4:00:00

# Load environment
module purge
module load cuda/11.8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate paramem

# Set paths
BASE_DIR="/home/mmahaut/projects/paramem"
CHECKPOINT_PATH="__CHECKPOINT_PATH__"
CHECKPOINT_TYPE="__CHECKPOINT_TYPE__"

# Determine model path based on type
if [ "$CHECKPOINT_TYPE" = "fullft" ]; then
    MODEL_PATH="${CHECKPOINT_PATH}/merged_single_model"
    echo "Loading Full-FT model from: ${MODEL_PATH}"
else
    MODEL_PATH="${CHECKPOINT_PATH}"
    echo "Loading LoRA adapter from: ${MODEL_PATH}"
fi

# Ensure slurm_logs directory exists
mkdir -p "${CHECKPOINT_PATH}/slurm_logs"

echo "==================================================================="
echo "Paraphrase Evaluation with LLM-Generated Paraphrases"
echo "==================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Type: ${CHECKPOINT_TYPE}"
echo "Model path: ${MODEL_PATH}"
echo "Start time: $(date)"
echo ""

# Check GPU availability
nvidia-smi

echo ""
echo "==================================================================="
echo "Running Paraphrase Evaluation"
echo "==================================================================="
echo ""

# Run evaluation - only the paraphrase part
cd "${BASE_DIR}"

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from paramem.evaluation.wikidata_paraphrase_eval import evaluate_wikidata_with_paraphrases
import json
import os

# Load model and tokenizer for evaluation
print('Loading evaluation model and tokenizer...')
checkpoint_type = '${CHECKPOINT_TYPE}'
model_path = '${MODEL_PATH}'

if checkpoint_type == 'fullft':
    # Load full fine-tuned model
    print(f'Loading Full-FT model from: {model_path}')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    # Load base model and LoRA adapter
    print(f'Loading base model: meta-llama/Llama-3.1-8B-Instruct')
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    print(f'Loading LoRA adapter from: {model_path}')
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge adapter for faster inference
    tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

# Load base model for paraphrasing (consistent across all checkpoints)
print('\\\\nLoading base model for paraphrasing...')
paraphrase_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
paraphrase_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
paraphrase_tokenizer.pad_token = paraphrase_tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Models loaded on device: {device}')

# Set paths
wikidata_path = '${BASE_DIR}/data3/wikidata_Mis7.csv'
output_file = '${CHECKPOINT_PATH}/slurm_logs/wikidata_paraphrase_llm_results.json'
cache_file = '${BASE_DIR}/data3/wikidata_paraphrase_llm_cache.json'

print(f'\\\\nEvaluating Wikidata with LLM paraphrases...')
print(f'Output file: {output_file}')
print(f'Cache file: {cache_file}')
print(f'Using base model for paraphrasing (consistent across checkpoints)')

# Run evaluation
results = evaluate_wikidata_with_paraphrases(
    wikidata_path, model, tokenizer, device,
    n_samples=200,
    num_paraphrases=5,
    use_llm_paraphrasing=True,  # Use LLM paraphrasing
    paraphrase_model=paraphrase_model,  # Use base model for paraphrasing
    paraphrase_tokenizer=paraphrase_tokenizer,
    output_file=output_file,
    paraphrase_cache_file=cache_file
)

print('\\\\n=== Paraphrase Evaluation Results ===')
for key, value in results.items():
    print(f'{key}: {value:.4f}')

# Save summary
summary_file = '${CHECKPOINT_PATH}/slurm_logs/paraphrase_llm_summary.json'
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\\\nSummary saved to: {summary_file}')
"

echo ""
echo "==================================================================="
echo "Evaluation Complete"
echo "End time: $(date)"
echo "==================================================================="
EOFSCRIPT

    # Replace placeholders - use different delimiter for sed to avoid issues with paths
    sed -i "s|__CHECKPOINT_PATH__|${ckpt_path}|g" "$job_script"
    sed -i "s|__CHECKPOINT_TYPE__|${ckpt_type}|g" "$job_script"
    
    # Submit job
    job_id=$(sbatch "$job_script" | awk '{print $4}')
    echo "  → Submitted job $job_id"
    
    # Clean up temp script
    rm "$job_script"
    
    echo ""
done

echo "==================================================================="
echo "All jobs submitted!"
echo "==================================================================="
echo ""
echo "To monitor jobs: squeue -u \$USER"
echo "To check results: tail -f models3/*/checkpoint-*/slurm_logs/paraphrase_eval_*.out"
