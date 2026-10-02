#!/bin/bash
# Wrapper script for launching fine-tuning with different GPU configurations
# Usage: 
#   ./launch_training.sh                    # Single GPU (default)
#   ./launch_training.sh 3                  # 3 GPUs, 1 node  
#   ./launch_training.sh 5                  # 5 GPUs, 1 node
#   ./launch_training.sh 10                 # 10 GPUs, 2 nodes (5 GPUs each)
#   ./launch_training.sh 16                 # 16 GPUs, 4 nodes (5 GPUs each -> 20 total)
#   NUM_GPUS=8 ./launch_training.sh         # 8 GPUs, 2 nodes (alternative syntax)

# Set NUM_GPUS from command line argument or environment variable
if [ $# -gt 0 ]; then
    export NUM_GPUS=$1
    echo "🚀 Launching training with $NUM_GPUS GPUs per node (from command line)"
elif [ ! -z "$NUM_GPUS" ]; then
    echo "🚀 Launching training with $NUM_GPUS GPUs per node (from environment)"
else
    export NUM_GPUS=1
    echo "🚀 Launching training with $NUM_GPUS GPU per node (default)"
fi

# Validate NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ Error: NUM_GPUS must be a positive integer, got: $NUM_GPUS"
    exit 1
fi

# Calculate nodes and GPUs per node (max 5 GPUs per node)
GPUS_PER_NODE=5
if [ "$NUM_GPUS" -le 5 ]; then
    # Single node case
    export NUM_NODES=1
    export GPUS_PER_NODE=$NUM_GPUS
else
    # Multi-node case: distribute GPUs across nodes
    export NUM_NODES=$(( (NUM_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))  # Ceiling division
    # Use full 5 GPUs per node for multi-node setups
    export GPUS_PER_NODE=5
fi

# Calculate total GPUs that will be used (might be slightly more than requested for optimal distribution)
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Show configuration
echo "📊 Configuration:"
echo "   - Requested GPUs: $NUM_GPUS"
echo "   - Nodes: $NUM_NODES"
echo "   - GPUs per node: $GPUS_PER_NODE"
echo "   - Total GPUs used: $TOTAL_GPUS"
if [ "$TOTAL_GPUS" -gt "$NUM_GPUS" ]; then
    echo "   ⚠️  Note: Using $TOTAL_GPUS GPUs (rounded up for optimal node distribution)"
fi
echo "   - Memory will be adjusted based on training type (LoRA vs Full FT)"
echo ""

# Launch the main training script
exec ./scripts/fine_tune_multi_node.sh