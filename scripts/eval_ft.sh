#!/bin/bash

# Parameters
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --gres=gpu:5
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
# models=("meta-llama/Meta-Llama-3.1-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "EleutherAI/pythia-6.9b")
models=("mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2-7B" "Qwen/Qwen2-7B-Instruct" "EleutherAI/pythia-6.9b")
# models=("EleutherAI/pythia-6.9b")
ckpts=("./models/Mis7-ft/checkpoint-4110/pytorch_model_fsdp_0" "./models/Mis7i-ft/checkpoint-3500/pytorch_model_fsdp_0" "./models/Qwe7-ft/checkpoint-3000/pytorch_model_fsdp_0" "./models/Qwe7i-ft/checkpoint-4110/pytorch_model_fsdp_0" "./models/pyt7-ft/checkpoint-3500"/pytorch_model_fsdp_0)
# ckpts=("./models/Qwe7-ft/")
# ckpts=("" "" "" "")
# launch sbatch with same parameters for each model

i=0
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
    # data_path="./data3/wikidata_Mis7.csv"
    # data_path=/home/mmahaut/projects/paramem/models/Mis7-ft/data.csv
    data_path=/home/mmahaut/projects/paramem/data/pile_$jobname.txt

    echo "TOKENIZERS_PARALLELISM=false poetry run python ./paramem/slot_filling.py \
    --dataset-path=$data_path \
    --save-inputs=./data3/ftkk_inputs_$jobname.csv \
    --batch-size=10\
    --input-key=query\
    --outpath=\"./data3/ftkk_$jobname.csv\" \
    --log-path=\"./logs/ftkk_$jobname\" \
    --model-name=\"$model\" \
    --checkpoint-path=\"${ckpts[$i]}\" \
    --num-return-sequences=1 " >> slurm.sh
    # echo "python /home/mmahaut/projects/parametric_mem/slot_filling.py test-generation --outpath=\"./data2/wikidata_$jobname.csv\" --log-path=\"./logs2/wikidata_$jobname\" --model-name=\"$model\"" >> slurm.sh
    sed -i "12s/^/#SBATCH --job-name=ftpile_$jobname\n/" slurm.sh
    ((i=i+1))
    sbatch slurm.sh
done