#!/bin/bash
# Load environment using shared helper
source /home/mmahaut/projects/paramem/scripts/entrypoints/_common_env.sh
paramem_activate_env 1
export CUDA_HOME=/soft/easybuild/x86_64/software/CUDA/12.1.0/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Test script to verify the training works with a small dataset
cd ~/projects/paramem/

# Test with a small dataset first - NO DeepSpeed for single GPU test
srun --ntasks=1 python /home/mmahaut/projects/paramem/paramem/fine_tune_with_ckpts.py \
  --model-name "mistralai/Mistral-7B-v0.3" \
  --dataset-name "/home/mmahaut/projects/paramem/data/book.txt" \
  --output-dir "./test_training_fix" \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --block-size 184 \
  --per-device-train-batch-size 1 \
  --num-train-epochs 1 \
  --save-steps 10 \
  --logging-steps 5 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --n-samples 50

echo "Test completed!"