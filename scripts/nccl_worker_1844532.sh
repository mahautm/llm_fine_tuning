#!/bin/bash
source ~/.bashrc
module load CUDA/12.1.0
conda activate paramem

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=2 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /home/mmahaut/projects/paramem/scripts/nccl_test_$SLURM_JOB_ID.py
