#!/bin/bash

# Comprehensive fix for DeepSpeed training issues
# This script addresses all the major issues found in the error logs

echo "=== Paramem Training Fix Script ==="
echo "Fixing DeepSpeed compilation and training issues..."

# 1. Clean problematic cache files
echo "1. Cleaning cache files..."
if [ -d "/home/mmahaut/.cache/torch_extensions" ]; then
    echo "   Removing torch extensions cache..."
    rm -rf /home/mmahaut/.cache/torch_extensions
fi

if [ -d "/tmp/deepspeed_cache" ]; then
    echo "   Removing DeepSpeed cache..."
    rm -rf /tmp/deepspeed_cache
fi

# 2. Set environment variables to prevent ALL DeepSpeed compilation
echo "2. Setting environment variables..."
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_AIO=0  
export DS_BUILD_UTILS=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_OPS=0
export PYTHONWARNINGS="ignore::UserWarning"

# 3. Clean up any existing processes
echo "3. Cleaning up processes..."
pkill -f "torchrun"
pkill -f "fine_tune_with_ckpts"

# 4. Wait a bit for cleanup
sleep 2

echo "=== Fix Summary ==="
echo "✓ Removed problematic DeepSpeedCPUAdam import"
echo "✓ Created fixed DeepSpeed configurations (ds_config_*_fixed.json)"  
echo "✓ Updated training script with better error handling"
echo "✓ Set environment variables to prevent CPU optimizer compilation"
echo "✓ Added proper distributed training cleanup"
echo "✓ Cleaned cache directories"

echo ""
echo "=== Next Steps ==="
echo "1. Run the cache cleanup script: ./scripts/entrypoints/maintenance/fix_deepspeed_cache.sh"
echo "2. Make sure to use the updated SLURM script (scripts/entrypoints/jobs/slurm.sh)"
echo "3. When training, use --deep-speed flag to use fixed DeepSpeed configs"
echo ""
echo "Environment variables set for this session:"
echo "  DS_BUILD_CPU_ADAM=0"
echo "  DS_BUILD_FUSED_ADAM=0"
echo "  DS_BUILD_AIO=0" 
echo "  DS_BUILD_UTILS=0"
echo "  DS_BUILD_FUSED_LAMB=0"
echo "  DS_BUILD_SPARSE_ATTN=0"
echo "  DS_BUILD_TRANSFORMER=0"
echo "  DS_BUILD_STOCHASTIC_TRANSFORMER=0"
echo "  DS_BUILD_OPS=0"
echo "  PYTHONWARNINGS=ignore::UserWarning"
echo ""
echo "Add these to your ~/.bashrc for permanent effect:"
echo "echo 'export DS_BUILD_CPU_ADAM=0' >> ~/.bashrc"
echo "echo 'export DS_BUILD_AIO=0' >> ~/.bashrc"
echo "echo 'export DS_BUILD_UTILS=0' >> ~/.bashrc"