#!/bin/bash
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --job-name=mp-partition-exp1
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_partition.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%j_mp_partition.err

set -euo pipefail

source ~/.bashrc
module load CUDA/12.1.0 || true
conda activate paramem

cd /home/mmahaut/projects/paramem

# Example dataset inputs; replace with the activation pickle paths you want to compare.
INPUT_1="pile=results/mp_reservoir/activations/pile_activations.pickle"
INPUT_2="biomed=results/mp_reservoir/activations/biomed_activations.pickle"

srun python scripts/analysis/run_mp_partition_stability.py \
  --input "$INPUT_1" \
  --input "$INPUT_2" \
  --layers "0,4,8,12,16,20,24" \
  --output-dir "results/mp_reservoir/spectra" \
  --tag "first_pass"
