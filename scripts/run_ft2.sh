#!/bin/bash
source ~/.bash_profile

# Unset mutually exclusive SLURM memory variables to avoid srun fatal error
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
# unset SLURM_MEM_PER_NODE

# --- Safety flags ---
set -euo pipefail

# --- Ensure critical variables are present ---
: "${MODEL_NAME:?Need to set MODEL_NAME}"
: "${DATASET_NAME:?Need to set DATASET_NAME}"
: "${CHECKPOINT_PATH:?Need to set CHECKPOINT_PATH}"
: "${USE_LORA:?Need to set USE_LORA}"
: "${MEM:?Need to set MEM}"
: "${PARTITION:?Need to set PARTITION}"
: "${LR:?Need to set LR}"

# --- Configure master address/port for rendezvous ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
# Use a deterministic rendezvous port per job to reduce startup races.
RDZV_PORT_BASE=${RDZV_PORT_BASE:-29400}
export MASTER_PORT=$(( RDZV_PORT_BASE + ( SLURM_JOB_ID % 1000 ) ))
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6,roundup_power2_divisions:16

echo "---------------------------------------------"
echo "Node      : $(hostname)"
echo "SLURM ID  : $SLURM_JOB_ID"
echo "Node List : $SLURM_NODELIST"
echo "Node ID   : $SLURM_NODEID"
echo "Model     : $MODEL_NAME"
echo "Dataset   : $DATASET_NAME"
echo "LoRA      : $USE_LORA"
echo "LR       : $LR"
echo "Master    : $MASTER_ADDR:$MASTER_PORT"
echo "---------------------------------------------"


srun /bin/hostname
srun /bin/python3 -c "import socket; print(socket.gethostname())"
srun --label --export=ALL /bin/bash /home/mmahaut/projects/paramem/scripts/run_ft_worker.sh