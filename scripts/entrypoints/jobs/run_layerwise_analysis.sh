#!/bin/bash
#SBATCH --job-name=layerwise-lora-full
#SBATCH --output=/home/mmahaut/projects/paramem/layerwise_%j.out
#SBATCH --error=/home/mmahaut/projects/paramem/layerwise_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
source /home/mmahaut/projects/paramem/scripts/entrypoints/_common_env.sh
paramem_activate_env 0
cd /home/mmahaut/projects/paramem
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-4} python scripts/analysis/analyze_layerwise_lora_full.py
