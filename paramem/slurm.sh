#!/bin/bash

# Parameters
#SBATCH --mem=100G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --error=/home/mmahaut/projects/exps/dl/%j_0_log.err
#SBATCH --job-name=extract-hlayer
#SBATCH --job-name=hlft-Met7i
#SBATCH --output=/home/mmahaut/projects/exps/dl/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/paramem/paramem/
python /home/mmahaut/projects/paramem/paramem/extract_hidden.py "meta-llama/Meta-Llama-3-8B-Instruct" 10 ./data/wikidata_Met7i.csv --input-key=context_query --out-pickle-prefix="./hlayer/Met7i-icl"
