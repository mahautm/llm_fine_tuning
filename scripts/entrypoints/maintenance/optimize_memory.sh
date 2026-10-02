#!/bin/bash

# Memory optimization script for large model training
# Run this before starting training to optimize CUDA memory settings

echo "=== Memory Optimization for Large Model Training ==="

# Set conservative CUDA memory management.
# Avoid expandable_segments due to observed allocator internal asserts on this cluster.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,roundup_power2_divisions:8

# Disable CUDA caching to reduce memory fragmentation
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=1

# Enable memory pool optimization
export PYTORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=16
export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=16

# Clear any existing CUDA memory
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Status Before Cleanup:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
    
    # Force garbage collection
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('CUDA cache cleared')
"
    
    echo "GPU Memory Status After Cleanup:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
fi

echo "Memory optimization settings applied!"
echo "Environment variables:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "  CUDA_CACHE_DISABLE=$CUDA_CACHE_DISABLE"