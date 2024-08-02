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
#SBATCH --job-name=hlft-t7i
#SBATCH --output=/home/mmahaut/projects/exps/dl/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/parametric_mem/
python /home/mmahaut/projects/parametric_mem/extract_hidden_v2.py "/home/mmahaut/projects/parametric_mem/models/t7i-ft/checkpoint-4500" 10 ./data/wikidata_incl_t7i.csv --data-file2=./data/wikidata_nli_t7i.csv --out-pickle-prefix="./hlayer/t7i-ft" --instruction="Accurately fill in the following sentence with the correct word, creating factual sentences as in the examples:" --examples="The author of The Homecoming is Harold Pinter." --examples="Marie Curie was born in Warsaw." --examples="Victor Hugo was born in France."
