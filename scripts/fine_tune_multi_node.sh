#!/bin/bash
# File: fine_tune_ckpt_eval.sh

# BUILD the pile dataset to have the same number of lines as the factual dataset
data_path="/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv"
num_lines=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$data_path')))")
data_path="/home/mmahaut/projects/paramem/data/pile_19_token_remaining_sequences.txt"
head -n $num_lines $data_path > "/home/mmahaut/projects/paramem/data/pile_19_short.txt"

echo "=== Job Script Running On Node: $(hostname) ==="
echo "=== Current Working Directory: $(pwd) ==="
echo "=== Applying DeepSpeed fixes before training ==="

# Apply DeepSpeed fixes
cd /home/mmahaut/projects/paramem
if [ -f "./scripts/entrypoints/maintenance/complete_fix.sh" ]; then
  echo "Running comprehensive fix script..."
  ./scripts/entrypoints/maintenance/complete_fix.sh
fi
if [ -f "./scripts/entrypoints/maintenance/fix_deepspeed_cache.sh" ]; then
  echo "Cleaning DeepSpeed cache..."
  ./scripts/entrypoints/maintenance/fix_deepspeed_cache.sh
fi

# Memory configuration based on training type
MEM_LORA="100G"      # Memory for LoRA training
MEM_FULL="180G"      # Increased memory for full fine-tuning
PARTITION="alien"
QOS="alien"
EXCLUDE_NODES="node044"

# GPU configuration per training type
# Full FT: 8 GPUs (2 nodes × 4 GPUs) - matches LoRA effective batch size
# LoRA: 4 GPUs (1 node × 4 GPUs) - increased for MMLU memory requirements
GPUS_FULL_FT=8
NODES_FULL_FT=2
GPUS_PER_NODE_FULL_FT=4

GPUS_LORA=4
NODES_LORA=1
GPUS_PER_NODE_LORA=4

BASE_DIR="/home/mmahaut/projects/paramem/models3"
# MODEL_NAMES=("mistralai/Mistral-7B-v0.3" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct")
# MODEL_NAMES=("mistralai/Mistral-7B-v0.3")
# MODEL_NAMES=("meta-llama/Llama-3.2-1B")
MODEL_NAMES=("meta-llama/Llama-3.1-8B-Instruct")
DATASET_NAMES=(
  # "/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv"
  # "/home/mmahaut/projects/paramem/data/pile_19_short.txt"
  "/home/mmahaut/projects/paramem/data/mmlu_train_short.txt"
)
LR=1e-4


for model in "${MODEL_NAMES[@]}"; do
  for USE_LORA in 0 1; do
    for dataset_path in "${DATASET_NAMES[@]}"; do
      SHORT_MODEL_NAME="${model#*/}"

      if [[ $(basename "$dataset_path") == "pile_19_short.txt" ]]; then
        DATASET_TAG="pile"
      elif [[ $(basename "$dataset_path") == "mmlu_train.txt" || $(basename "$dataset_path") == "mmlu_train_short.txt" ]]; then
        DATASET_TAG="mmlu"
      else
        DATASET_TAG="wikiplus"
      fi
      CHECKPOINT_PATH="${BASE_DIR}/${SHORT_MODEL_NAME}-fsdp-${USE_LORA}-${DATASET_TAG}-lr${LR}"
      mkdir -p "$CHECKPOINT_PATH/slurm_logs"

      # Set memory and GPU configuration based on training type
      if [[ $USE_LORA -eq 1 ]]; then
        MEM=$MEM_LORA
        NUM_NODES=$NODES_LORA
        GPUS_PER_NODE=$GPUS_PER_NODE_LORA
        TOTAL_GPUS=$GPUS_LORA
        echo "LoRA training: Using $MEM memory, $NUM_NODES node(s), $GPUS_PER_NODE GPU(s) per node = $TOTAL_GPUS total GPUs"
      else
        MEM=$MEM_FULL
        NUM_NODES=$NODES_FULL_FT
        GPUS_PER_NODE=$GPUS_PER_NODE_FULL_FT
        TOTAL_GPUS=$GPUS_FULL_FT
        echo "Full fine-tuning: Using $MEM memory, $NUM_NODES nodes, $GPUS_PER_NODE GPUs per node = $TOTAL_GPUS total GPUs"
      fi
      
      GRES="gpu:${GPUS_PER_NODE}"

      echo "Launching job for model=$model, dataset=$dataset_path"

      export MODEL_NAME="$model"
      export DATASET_NAME="$dataset_path"
      export CHECKPOINT_PATH="$CHECKPOINT_PATH"
      export USE_LORA="$USE_LORA"
      export MEM
      export PARTITION
      export QOS
      export EXCLUDE_NODES
      export GRES
      export TOTAL_GPUS
      export NUM_NODES
      export GPUS_PER_NODE
      export LR
      export KEEP_CHECKPOINT=1  # Preserve checkpoints after evaluation
      echo "MODEL_NAME=$MODEL_NAME DATASET_NAME=$DATASET_NAME CHECKPOINT_PATH=$CHECKPOINT_PATH USE_LORA=$USE_LORA MEM=$MEM PARTITION=$PARTITION QOS=$QOS EXCLUDE_NODES=$EXCLUDE_NODES GRES=$GRES NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE TOTAL_GPUS=$TOTAL_GPUS"
      sbatch \
        --mem=$MEM \
        --partition=$PARTITION \
        --qos=$QOS \
        --exclude=$EXCLUDE_NODES \
        --gres=$GRES \
        --output=${CHECKPOINT_PATH}/slurm_logs/%j_%N.out \
        --error=${CHECKPOINT_PATH}/slurm_logs/%j_%N.err \
        --nodes=$NUM_NODES \
        --spread-job \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --export=ALL \
        --job-name="ft-${SHORT_MODEL_NAME}-${DATASET_TAG}-${USE_LORA}-${NUM_NODES}n" \
        /home/mmahaut/projects/paramem/scripts/run_ft2.sh
    done
  done
done
