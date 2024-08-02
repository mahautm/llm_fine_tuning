#!/bin/bash

# Parameters
#SBATCH --mem=164G
#SBATCH --partition=alien
#SBATCH --gres=gpu:2
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --error=/home/mmahaut/projects/exps/sf/%j_0_log.err
#SBATCH --job-name=sf_launcher
#SBATCH --output=/home/mmahaut/projects/exps/sf/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/

cd ~/projects/paramem/
models=("mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct")
# models=("EleutherAI/pythia-6.9b" "facebook/opt-6.7b")
# launch sbatch with same parameters for each model
for model in "${models[@]}"
do
    # Set jobname variable
    if [[ $model == *"nstruct"* ]]; then
        jobname="${model#*/}"
        jobname="${jobname:0:3}7i"
    else
        jobname="${model#*/}"
        jobname="${jobname:0:3}7"
    fi

    echo "Launching for model $model and nli $nli with jobname $jobname"
    current_path=$(realpath "$0")
    echo "Current file path: $current_path"
    head -n 21 $current_path > slurm.sh
    data_path="./data/wikidata_$jobname"".csv"
    echo "TOKENIZERS_PARALLELISM=false poetry run python ./paramem/slot_filling.py test-generation \
    --only-unknowns \
    --dataset-path=$data_path \
    --save-inputs=./data/sf2_inputs_$jobname.csv \
    --input-key=context_query\
    --outpath=\"./data/cont_wikidata_$jobname.csv\" \
    --log-path=\"./logs/wikidata_cont_$jobname\" \
    --model-name=\"$model\" \
    --num-return-sequences=1" >> slurm.sh

    sed -i "12s/^/#SBATCH --job-name=sf_$jobname\n/" slurm.sh

    sbatch slurm.sh
done