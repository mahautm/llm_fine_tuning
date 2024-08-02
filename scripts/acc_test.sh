#!/bin/bash

# Parameters
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --error=/home/mmahaut/projects/exps/acc/%j_0_log.err
#SBATCH --job-name=slot_filling
#SBATCH --output=/home/mmahaut/projects/exps/acc/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/

cd ~/projects/parametric_mem/
# models=("mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.2" "tiiuae/falcon-7b" "tiiuae/falcon-7b-instruct")
models=("/home/mmahaut/projects/parametric_mem/models/m7-ft/checkpoint-390000" "/home/mmahaut/projects/parametric_mem/models/m7i-ft/checkpoint-430000" "/home/mmahaut/projects/parametric_mem/models/t7-ft/checkpoint-350000" "/home/mmahaut/projects/parametric_mem/models/t7i-ft/checkpoint-450000")

# launch sbatch with same parameters for each model
for model in "${models[@]}"
do
    for nli in True False
    do
        # # Set jobname variable
        # if [[ $model == *"nstruct"* ]]; then
        #     jobname="${model:0:1}7i"
        # else
        #     jobname="${model:0:1}7"
        # fi

        # if [[ $nli == "True" ]]; then
        #     jobname="pr_nli_$jobname"
        # else
        #     jobname="pr_incl_$jobname"
        # fi
        jobname=$(dirname $model | xargs basename | cut -d'-' -f1 | cut -d'/' -f2 | cut -d'.' -f1)
        prefix="wikidata"

        echo "Launching for model $model and nli $nli with jobname $jobname"
        current_path=$(realpath "$0")
        echo "Current file path: $current_path"
        head -n 21 $current_path > slurm.sh
        data_path="./data/$prefix""_incl_$jobname.csv"
        additional_data_path="./data/$prefix""_nli_$jobname.csv"
        echo "python /home/mmahaut/projects/parametric_mem/slot_filling.py check-acc \"$model\" $data_path --n-samples=10 --data-file2=\"$additional_data_path\" --outpath=\"./data/acc_$jobname.csv\" --instruction=\"Accurately fill in the following sentence with the correct word, creating factual sentences as in the examples:\" --examples=\"The author of The Homecoming is Harold Pinter.\" --examples=\"Marie Curie was born in Warsaw.\" --examples=\"Victor Hugo was born in France.\" $([[ $nli == "True" ]] && echo "--use-nli")" >> slurm.sh
        sed -i "12s/^/#SBATCH --job-name=ACC_$jobname\n/" slurm.sh


        sbatch slurm.sh

    done
done
