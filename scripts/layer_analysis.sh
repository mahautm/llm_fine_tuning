#!/bin/bash

# Parameters
#SBATCH --mem=100G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --error=/home/mmahaut/projects/exps/la/%j_0_log.err
#SBATCH --job-name=Manova
#SBATCH --output=/home/mmahaut/projects/exps/la/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/paramem/
poetry run python ./paramem/layer_analysis.py
