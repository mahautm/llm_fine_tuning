#!/bin/bash
#SBATCH --job-name=llama7b-full-ft
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=250GB
#SBATCH --time=72:00:00
#SBATCH --exclude=node044
#SBATCH --output=slurm_logs/llama7b_full_ft_%j.out
#SBATCH --error=slurm_logs/llama7b_full_ft_%j.err

# Full fine-tuning of Llama-3.1-8B with FSDP on 2 GPUs

echo "=========================================="
echo "Llama-3.1-8B Full Fine-Tuning with FSDP"
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
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="./data/pile_train.txt"  # Change to your dataset
OUTPUT="./models2/Llama-3.1-8B-full-ft-pile-lr1e-5"
LR="1e-5"
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
NUM_GPUS=2

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE per device"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  Training Mode: Full Fine-Tuning (FSDP)"
echo ""

cd /home/mmahaut/projects/paramem

# Run training with torchrun for distributed setup
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 \
  paramem/fine_tune_with_ckpts.py \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --output-dir "$OUTPUT" \
  --per-device-train-batch-size $BATCH_SIZE \
  --gradient-accumulation-steps $GRAD_ACCUM \
  --learning-rate $LR \
  --num-train-epochs $EPOCHS \
  --save-steps 500 \
  --logging-steps 10 \
  --eval-steps 500 \
  --fsdp \
  --gradient-checkpointing \
  --max-grad-norm 1.0 \
  --bf16 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
