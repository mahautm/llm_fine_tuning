#!/bin/bash
# launch_memorization_only_tests.sh
# Modified version: only runs memorization evaluation, skips performance/ID
# Deletes checkpoints after evaluation (except every 3rd checkpoint per epoch)
# Target: models4 directory with wikiplus only

# Exclude problematic nodes
EXCLUDE_NODES="node044,node041"

# Script paths
MEM_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/memorization_evaluation.py"

# Output directories
OUT_BASE="/home/mmahaut/projects/paramem/models4"
mkdir -p "$OUT_BASE"

# Models to train
MODELS=(
    "Llama-3.1-8B-Instruct"
)

DATASETS=(
    "wikiplus"
)

TRAINING_TYPES=(
    "fsdp-0"  # Full-FT
    "fsdp-1"  # LoRA
)

echo "=========================================="
echo "Memorization-Only Training Pipeline"
echo "Target: models4 (Wikiplus only)"
echo "=========================================="
echo ""

# For each training type and dataset
for TRAINING_TYPE in "${TRAINING_TYPES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        MODEL_OUTPUT="${OUT_BASE}/${MODELS[0]}-${TRAINING_TYPE}-${DATASET}-lr1e-4"
        USE_LORA=$([ "$TRAINING_TYPE" = "fsdp-1" ] && echo 1 || echo 0)
        
        echo "Setting up: ${MODELS[0]} ${TRAINING_TYPE} ${DATASET}"
        echo "  Output: $MODEL_OUTPUT"
        echo "  LoRA: $USE_LORA"
        echo ""
        
        mkdir -p "$MODEL_OUTPUT"
        
        # Here you would add the training command
        # This is a placeholder showing the structure
        echo "TODO: Add training command for $MODEL_OUTPUT"
        
    done
done

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Add actual training commands above"
echo "2. Run this script to start training"
echo "3. Monitor with: squeue -u \$USER | grep wikiplus"
echo "=========================================="
