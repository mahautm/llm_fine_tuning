#!/bin/bash

# Parameters
#SBATCH --mem=200G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/exps/paramem/%j_0_log.err
#SBATCH --job-name=extract-hlayer3
#SBATCH --output=/home/mmahaut/projects/exps/paramem/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate paramem
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/paramem/
models=("mistralai/Mistral-7B-v0.3") # right now the code only works with one model at a time

# don't put empty checkpoints first, as they have no data and will be using data from the first checkpoint
# ckpts=("./models/Mis7-ft/checkpoint-4110/pytorch_model_fsdp_0" "/home/mmahaut/projects/paramem/models/Mis7-pileft/checkpoint-1746/pytorch_model_fsdp_0")
ckpts=("./models/Mis7-ft/checkpoint-4110/pytorch_model_fsdp_0" "")
# ckpts=("/home/mmahaut/projects/paramem/models/Mis7-lora-ft/lora_weights.pth" "/home/mmahaut/projects/paramem/models/Mis7-lora-pileft/lora_weights.pth")
execute_slurm() {
    sed -i "12s/^/#SBATCH --job-name=hlft-$jobname\n/" slurm.sh
    sbatch slurm.sh
    rm slurm.sh
    echo "Launched job $jobname, reinitializing slurm.sh"
    head -n 21 $current_path > slurm.sh
}

i=0
for model in "${models[@]}"
do
    for checkpoint in "${ckpts[@]}"
    do
        echo checkpoint: $checkpoint
        # Set jobname variable
        if [[ $model == *"nstruct"* ]]; then
            jobname="${model#*/}"
            jobname="${jobname:0:3}7i"
        else
            jobname="${model#*/}"
            jobname="${jobname:0:3}7"
        fi

        if [[ $checkpoint != "" ]]; then
            jobname="${jobname}-ft"
        fi

        if [[ $checkpoint == *"lora"* ]]; then
            jobname="${jobname}-lora"
        fi

        if [[ $checkpoint == *"pile"* ]]; then
            jobname="${jobname}-pile"
        fi

        echo "Launching fine-tuning for model $model with jobname $jobname"
        current_path=$(realpath "$0")
        echo "Current file path: $current_path"
        head -n 21 $current_path > slurm.sh

        if [[ $checkpoint != "" ]]; then

            # train_data=$(dirname $(dirname $checkpoint))/train.csv
            train_data=$(dirname $checkpoint)/train.csv
            # make it into a txt file, skipping the first line
            tail -n +2 $train_data > ${train_data%.*}.txt
            train_data=${train_data%.*}.txt
            echo "poetry run python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $train_data --checkpoint-path=\"$checkpoint\" --out-pickle-prefix=\"./hlayer/$jobname-train\"" >> slurm.sh
            execute_slurm

            # test_data=$(dirname $(dirname $checkpoint))/test.csv
            test_data=$(dirname $checkpoint)/test.csv
            tail -n +2 $test_data > ${test_data%.*}.txt
            test_data=${test_data%.*}.txt
            echo "poetry run python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $test_data --checkpoint-path=\"$checkpoint\" --out-pickle-prefix=\"./hlayer/$jobname-test\"" >> slurm.sh
            execute_slurm
            
        else
            data_path=$(dirname $(dirname $ckpts[0]))/train.csv
            tail -n +2 $data_path > ${data_path%.*}.txt
            data_path=${data_path%.*}.txt
            echo "poetry run python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --out-pickle-prefix=\"./hlayer/$jobname-train\"" >> slurm.sh
            execute_slurm
        fi

        # pile_data_9=/home/mmahaut/projects/paramem/data/pile_9_token_sample.txt
        # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $pile_data_9 --override-data-file --checkpoint-path=\"$checkpoint\" --out-pickle-prefix=\"./hlayer3/$jobname-pile8-ft\"" >> slurm.sh
        # execute_slurm

        if [[ $checkpoint != *"pile"* ]]; then
            pile_data_19=/home/mmahaut/projects/paramem/data/pile_19_token_sample.txt
            echo "poetry run python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $pile_data_19 --checkpoint-path=\"$checkpoint\" --out-pickle-prefix=\"./hlayer/$jobname-pile8\"" >> slurm.sh
            execute_slurm
        fi
        ((i=i+1))
    done
done
