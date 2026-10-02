#!/bin/bash
#SBATCH --job-name=V1_extract_vision
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out

source ~/.bashrc
conda activate paramem
python scripts/data/extract_vision_only.py
