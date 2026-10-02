#!/bin/bash
# Rerun failed evaluations for all 8B LoRA checkpoints

CHECKPOINTS=(
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-100"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-200"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-300"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-400"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-500"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-600"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-700"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-800"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-900"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1000"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1100"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1200"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1300"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1400"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1500"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-100"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-200"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-300"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-400"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-500"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-600"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-700"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-800"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-900"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1000"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1100"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1200"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1300"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1400"
"/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1500"
)

echo "Launching evaluation jobs for ${#CHECKPOINTS[@]} checkpoints..."

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "Launching: $ckpt"
  bash /home/mmahaut/projects/paramem/scripts/launch_checkpoint_tests_hand.sh "$ckpt" 1
  sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo "All evaluation jobs submitted!"
