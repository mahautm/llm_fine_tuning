#!/bin/bash

# Parameters
#SBATCH --mem=200G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/exps/dl/%j_0_log.err
#SBATCH --job-name=extract-hlayer2
#SBATCH --output=/home/mmahaut/projects/exps/dl/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate py39
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/paramem/
# models=("meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-v0.3")
# ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model.bin" "./models/Met7i-ft/checkpoint-4000/pytorch_model.bin" "./models/Mis7-ft/checkpoint-2000/pytorch_model.bin")
# ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/Met7i-ft/checkpoint-4000/pytorch_model_fsdp_0" "./models/Mis7-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/Mis7i-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/OLM7-ft/checkpoint-2000/pytorch_model_fsdp_0")
# ckpts=("./models/Met7-pileft/checkpoint-6000/pytorch_model_fsdp_0" "./models/Met7i-pileft/checkpoint-6000/pytorch_model_fsdp_0" "./models/Mis7-pileft/checkpoint-6000/pytorch_model_fsdp_0")
models=("meta-llama/Meta-Llama-3-8B")
ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model.bin")

# models=("allenai/OLMo-1.7-7B-hf")
# ckpts=("./models/OLM7-ft/checkpoint-2000/pytorch_model.bin")

execute_slurm() {
    sed -i "12s/^/#SBATCH --job-name=probe-$model_name\n/" slurm.sh
    sbatch slurm.sh
}

i=0
for model in "${models[@]}"
do
    current_path=$(realpath "$0")
    echo "Current file path: $current_path"
    head -n 21 $current_path > slurm.sh
    # find all the files in the data folder
    data_path=./data/probe_datasets
    data_files=$(ls $data_path)
    # data_files=("/home/mmahaut/projects/paramem/data/probe_datasets/sentence_length.txt")
    for data_file in $data_files
    do 
        # result file in format data_model_reps use basename
        stem=$(basename $data_file)
        model_name=$(basename $model)
        model_name=${model_name%%-*}
        if [[ $model_name == *"Meta"* ]]; then
            model_name="llama"
        fi
        if [[ $model == *"nstruct"* ]]; then
            model_name="${model_name}_instruct"
        fi
        result_file="${stem%.*}_${model_name}_reps"
        result_file=$(echo "$result_file" | tr '[:upper:]' '[:lower:]')
        echo "Launching fine-tuning for model $model with data file $data_file"
        # echo "poetry run python ./paramem/probing_task_extract_final_representations.py $model 10 $data_path/$data_file ./probe_representations/$result_file" >> slurm.sh
        result_file="${stem%.*}_${model_name}_pft_reps"
        result_file=$(echo "$result_file" | tr '[:upper:]' '[:lower:]')
        echo "poetry run python ./paramem/probing_task_extract_final_representations.py $model 10 $data_path/$data_file ./probe_representations/$result_file ${ckpts[$i]}" >> slurm.sh
        execute_slurm
        ((i++))
    done
done