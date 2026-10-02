#!/bin/bash

# Parameters
#SBATCH --mem=200G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/%j_node%N.err
#SBATCH --job-name=paramem-training
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/%j_node%N.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate paramem
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/

# Fix DeepSpeed compilation issues - disable ALL extensions
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_OPS=0

# Prevent some PyTorch warnings
export PYTHONWARNINGS="ignore::UserWarning"

cd ~/projects/paramem/
