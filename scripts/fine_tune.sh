#!/bin/bash

# Parameters
#SBATCH --mem=500G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:5
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/exps/ft/%j_0_log.err
#SBATCH --job-name=fine-tune
#SBATCH --output=/home/mmahaut/projects/exps/ft/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
module load CUDA/12.1.0
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/paramem/
# models=("mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "allenai/OLMo-1.7-7B-hf")
models=("EleutherAI/pythia-6.9b")
# models=("mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B")

prefix="wikidata"
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

    echo "Launching fine-tuning for model $model with jobname $jobname"
    current_path=$(realpath "$0")
    echo "Current file path: $current_path"
    head -n 21 $current_path > slurm.sh
    data_path="./data/wikidata_$jobname"".csv"

    # IF TRAINING ON PILE
    # get number of lines in file
    # num_lines=$(wc -l < $data_path)
    # data_path="./data/pile_19_token_remaining_sequences.txt"
    # data_path="./data/pile_9_token_remaining_sequences.txt"
    # head -n $num_lines $data_path > "./data/pile_$jobname"".txt"
    # data_path="./data/pile_$jobname"".txt"
    ## ADD pile to output dir: $jobname-pileft

    echo "echo $jobname && WANDB_WATCH=all PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false poetry run accelerate launch \
        ./paramem/fine_tune.py \
        --batch-size=5 \
        --epochs=5 \
        --model-name=\"$model\" \
        --train-file=$data_path \
        --save-inputs \
        --output-dir=\"./models/$jobname-ft\"" >> slurm.sh
    sed -i "12s/^/#SBATCH --job-name=sfft-$jobname\n/" slurm.sh
    sbatch slurm.sh
done