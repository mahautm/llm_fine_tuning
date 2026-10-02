#!/bin/bash
#SBATCH --job-name=H_extra_pile
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node044
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out

source ~/.bashrc
conda activate paramem
python scripts/data/extract_pile_only.py
