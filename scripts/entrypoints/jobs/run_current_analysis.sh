#!/bin/bash
#SBATCH --job-name=current-models2-analysis
#SBATCH --output=/home/mmahaut/projects/paramem/current_analysis_%j.out
#SBATCH --error=/home/mmahaut/projects/paramem/current_analysis_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00

# Load environment
source /home/mmahaut/projects/paramem/scripts/entrypoints/_common_env.sh
paramem_activate_env 0

# Change to project directory
cd /home/mmahaut/projects/paramem

# Run the comprehensive analysis
echo "Starting comprehensive analysis of current models2 directory..."
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-2} python scripts/analysis/analyze_current_models2.py

echo "Analysis completed!"
