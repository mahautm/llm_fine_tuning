#!/bin/bash
# Launch wikiplus training runs for models4 via run_ft2.sh (same pipeline as models3)
# Saves checkpoints every 130 steps (~3 per epoch, ~15 total over 5 epochs)

set -euo pipefail

EXCLUDE_NODES="node044,node041"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv"
DATASET_TAG="wikiplus"
OUTPUT_BASE="/home/mmahaut/projects/paramem/models4"
LR="1e-4"

# Memory / GPU config (mirrors fine_tune_multi_node)
MEM_LORA="100G"
MEM_FULL="400G"
PARTITION="alien"
QOS="alien"
GPUS_FULL_FT=8    # 2 nodes × 4 GPUs
NODES_FULL_FT=2
GPUS_PER_NODE_FULL_FT=4

GPUS_LORA=4       # 1 node × 4 GPUs
NODES_LORA=1
GPUS_PER_NODE_LORA=4

EVAL_SCRIPT="/home/mmahaut/projects/paramem/scripts/launch_memorization_checkpoint_eval.sh"
# Allow env override so we can temporarily keep checkpoints when debugging evals
KEEP_CHECKPOINT=${KEEP_CHECKPOINT:-0}   # cleanup after memorization eval

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Models4 Wikiplus Training Pipeline (run_ft2)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Learning Rate: $LR"
echo "Save Steps: 130 (~3 per epoch)"
echo "Exclude Nodes: $EXCLUDE_NODES"
echo "EVAL_SCRIPT: $EVAL_SCRIPT"
echo "KEEP_CHECKPOINT: $KEEP_CHECKPOINT"
echo ""

launch_job() {
  local USE_LORA=$1
  local MEM=$2
  local NODES=$3
  local GPUS_PER_NODE=$4
  local TOTAL_GPUS=$5
  local JOB_NAME=$6
  local OUTPUT_DIR=$7

  mkdir -p "$OUTPUT_DIR/slurm_logs"

  echo "Launching job: $JOB_NAME"
  echo "  USE_LORA=$USE_LORA MEM=$MEM NODES=$NODES GPUS_PER_NODE=$GPUS_PER_NODE TOTAL_GPUS=$TOTAL_GPUS"

  sbatch \
    --mem=$MEM \
    --partition=$PARTITION \
    --qos=$QOS \
    --exclude=$EXCLUDE_NODES \
    --gres=gpu:${GPUS_PER_NODE} \
    --output=${OUTPUT_DIR}/slurm_logs/%j_%N.out \
    --error=${OUTPUT_DIR}/slurm_logs/%j_%N.err \
    --nodes=$NODES \
    --spread-job \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --export=ALL,MODEL_NAME=$BASE_MODEL,DATASET_NAME=$DATASET,CHECKPOINT_PATH=$OUTPUT_DIR,USE_LORA=$USE_LORA,MEM=$MEM,PARTITION=$PARTITION,QOS=$QOS,EXCLUDE_NODES=$EXCLUDE_NODES,GRES=gpu:${GPUS_PER_NODE},TOTAL_GPUS=$TOTAL_GPUS,NUM_NODES=$NODES,GPUS_PER_NODE=$GPUS_PER_NODE,LR=$LR,KEEP_CHECKPOINT=$KEEP_CHECKPOINT,EVAL_SCRIPT=$EVAL_SCRIPT \
    --job-name="$JOB_NAME" \
    /home/mmahaut/projects/paramem/scripts/run_ft2.sh
}

# Full fine-tuning
OUTPUT_FULL="${OUTPUT_BASE}/Llama-3.1-8B-Instruct-fsdp-0-${DATASET_TAG}-lr${LR}"
launch_job 0 $MEM_FULL $NODES_FULL_FT $GPUS_PER_NODE_FULL_FT $GPUS_FULL_FT "models4-fsdp0-${DATASET_TAG}" "$OUTPUT_FULL"

# LoRA
OUTPUT_LORA="${OUTPUT_BASE}/Llama-3.1-8B-Instruct-fsdp-1-${DATASET_TAG}-lr${LR}"
launch_job 1 $MEM_LORA $NODES_LORA $GPUS_PER_NODE_LORA $GPUS_LORA "models4-fsdp1-${DATASET_TAG}" "$OUTPUT_LORA"

echo "=========================================="
echo "Training jobs launched via run_ft2.sh"
echo "Monitor with: watch squeue --me"
echo "=========================================="
