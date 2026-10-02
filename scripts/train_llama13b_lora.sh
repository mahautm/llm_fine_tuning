#!/bin/bash
#SBATCH --job-name=llama13b-lora
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300GB
#SBATCH --time=72:00:00
#SBATCH --exclude=node044
#SBATCH --output=slurm_logs/llama13b_lora_%j.out
#SBATCH --error=slurm_logs/llama13b_lora_%j.err

# Fine-tune Llama-2-13B with LoRA
# Requires 4 GPUs for memory efficiency

echo "=========================================="
echo "Llama-2-13B LoRA Fine-Tuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Activate environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate paramem

# Configuration
MODEL="meta-llama/Llama-2-13b-hf"
DATASET="./data/wikidata_pile.csv"  # Change to your dataset
OUTPUT="./models2/Llama-2-13B-lora-pile-lr5e-5"
LR="5e-5"  # Lower LR for bigger models
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LORA_R=16
LORA_ALPHA=32

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE per device"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM * 4)) (4 GPUs)"
echo "  LoRA rank: $LORA_R"
echo ""

cd /home/mmahaut/projects/paramem

# Run training with torchrun for distributed setup
torchrun --nproc_per_node=4 --nnodes=1 \
  paramem/fine_tune_with_ckpts.py \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --output-dir "$OUTPUT" \
  --use-lora \
  --lora-r $LORA_R \
  --lora-alpha $LORA_ALPHA \
  --lora-dropout 0.1 \
  --per-device-train-batch-size $BATCH_SIZE \
  --gradient-accumulation-steps $GRAD_ACCUM \
  --learning-rate $LR \
  --num-train-epochs $EPOCHS \
  --save-steps 500 \
  --logging-steps 10 \
  --fsdp \
  --max-grad-norm 1.0

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
