#!/bin/bash
#SBATCH --job-name=lora-fullft-analysis
#SBATCH --output=/home/mmahaut/projects/paramem/analysis_%j.out
#SBATCH --error=/home/mmahaut/projects/paramem/analysis_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Load environment
source /home/mmahaut/projects/paramem/scripts/entrypoints/_common_env.sh
paramem_activate_env 0

# Change to project directory
cd /home/mmahaut/projects/paramem

# Run the analysis
echo "Starting LoRA vs Full Fine-tuning comparison analysis..."
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-4} python scripts/analysis/analyze_lora_vs_full_ft.py

echo "Analysis completed!"
