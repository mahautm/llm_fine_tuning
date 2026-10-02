#!/bin/bash
# Test script for memory-optimized full fine-tuning

cd /home/mmahaut/projects/paramem

# Load environment using shared helper
source /home/mmahaut/projects/paramem/scripts/entrypoints/_common_env.sh
paramem_activate_env 1

echo "🧪 Testing memory-optimized full fine-tuning..."

# CUDA_HOME will be set by the module, but ensure it's in PATH
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

cd /home/mmahaut/projects/paramem

srun --ntasks=1 python paramem/fine_tune_with_ckpts.py \
    --model_name="mistralai/Mistral-7B-v0.3" \
    --data_file="./data/book.txt" \
    --output_dir="./test_full_ft_memory" \
    --max_length=64 \
    --batch_size=1 \
    --num_epochs=1 \
    --save_steps=10 \
    --logging_steps=5 \
    --max_steps=5 \
    --learning_rate=1e-4 \
    --deepspeed_config="./scripts/ds_config_full_conservative.json"

srun --ntasks=1 poetry run python paramem/fine_tune_with_ckpts.py \
  --model-name mistralai/Mistral-7B-v0.3 \
  --dataset-name /home/mmahaut/projects/paramem/data/book.txt \
  --output-dir ./test_full_ft_memory \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-train-epochs 1 \
  --save-steps 10 \
  --learning-rate 1e-4 \
  --n-samples 20 \
  --deep-speed \
  --logging-steps 5

echo "✅ Full fine-tuning memory test completed!"