#!/bin/bash

# Script to clean DeepSpeed cache and fix compilation issues
# This addresses the shared library import errors

echo "Cleaning DeepSpeed cache and torch extensions..."

# Remove cached torch extensions that are causing issues
if [ -d "/home/mmahaut/.cache/torch_extensions" ]; then
    echo "Removing torch extensions cache..."
    rm -rf /home/mmahaut/.cache/torch_extensions
fi

# Remove DeepSpeed cache if it exists
if [ -d "/tmp/deepspeed_cache" ]; then
    echo "Removing DeepSpeed cache..."
    rm -rf /tmp/deepspeed_cache
fi

# Set environment variables to prevent CPU Adam compilation
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0

echo "Cache cleanup complete!"
echo "Environment variables set:"
echo "  DS_BUILD_CPU_ADAM=0"
echo "  DS_BUILD_AIO=0" 
echo "  DS_BUILD_UTILS=0"
echo ""
echo "Please run the following before your next training:"
echo "export DS_BUILD_CPU_ADAM=0"
echo "export DS_BUILD_AIO=0"
echo "export DS_BUILD_UTILS=0"