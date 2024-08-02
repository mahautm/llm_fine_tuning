#!/bin/bash

# Parameters
#SBATCH --mem=164G
#SBATCH --partition=alien
#SBATCH --gres=gpu:3
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --error=/home/mmahaut/projects/exps/le/%j_0_log.err
#SBATCH --job-name=le_launcher
#SBATCH --output=/home/mmahaut/projects/exps/le/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python

cd ~/projects/paramem/
# models=("mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct")
models=("allenai/OLMo-1.7-7B-hf")
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

    echo "Launching for model $model with jobname $jobname"
    current_path=$(realpath "$0")
    echo "Current file path: $current_path"
    head -n 21 $current_path > slurm.sh
    echo "poetry run torchrun --standalone --nnodes=1 --nproc-per-node=3 -m tuned_lens train \
     --model.name $model \
     --data.name NeelNanda/pile-10k \
     --split train \
     --per_gpu_batch_size=1 \
     --num_steps=55 \
     --output my_lenses/$jobname \
     --fsdp" >> slurm.sh

    sed -i "12s/^/#SBATCH --job-name=le_$jobname\n/" slurm.sh

    sbatch slurm.sh
done