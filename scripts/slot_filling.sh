#!/bin/bash

# Parameters
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --gres=gpu:3
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
# models=("mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "tiiuae/falcon-7b" "tiiuae/falcon-7b-instruct" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct")
# models=("EleutherAI/pythia-6.9b")
models=("meta-llama/Meta-Llama-3.1-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "EleutherAI/pythia-6.9b")
# ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model.bin" "./models/Met7i-ft/checkpoint-4000/pytorch_model.bin" "./models/Mis7-ft/checkpoint-2000/pytorch_model.bin")
# ckpts=("./models/Met7-pileft/checkpoint-6000/pytorch_model.bin" "./models/Met7i-pileft/checkpoint-6000/pytorch_model.bin" "./models/Mis7-pileft/checkpoint-6000/pytorch_model.bin")

# ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/Met7i-ft/checkpoint-4000/pytorch_model_fsdp_0" "./models/Mis7-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/Mis7i-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/OLM7-ft/checkpoint-2000/pytorch_model_fsdp_0")

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
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --save-inputs=./data2/sf_nc_$jobname.csv --checkpoint-path=\"${ckpts[$i]}\" --outpath=\"./data2/wikidata_nc_$jobname.csv\" --log-path=\"./logs2/wikidata_$jobname\" --model-name=\"$model\" --num-return-sequences=10" >> slurm.sh
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --batch-size=8 --dataset-path=./data/pile_9_token_sample.txt --save-inputs=./data2/sf_pile8_$jobname.csv --outpath=\"./data2/pile8_$jobname.csv\" --log-path=\"./logs2/pile8_$jobname\" --model-name=\"$model\" --num-return-sequences=1 --collect-entropy" >> slurm.sh
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --batch-size=8 --dataset-path=./data/pile_19_token_sample.txt --save-inputs=./data2/sf_pile18_$jobname.csv --outpath=\"./data2/pile18_$jobname.csv\" --log-path=\"./logs2/pile18_$jobname\" --model-name=\"$model\" --num-return-sequences=1 --collect-entropy" >> slurm.sh
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --batch-size=8 --checkpoint-path=\"${ckpts[$i]}\" --dataset-path=./data/pile_9_token_sample.txt --save-inputs=./data2/sfft_pile8_$jobname.csv --outpath=\"./data2/pile8ft_$jobname.csv\" --log-path=\"./logs2/pile8ft_$jobname\" --model-name=\"$model\" --num-return-sequences=1 --collect-entropy">> slurm.sh
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --batch-size=8 --checkpoint-path=\"${ckpts[$i]}\" --dataset-path=./data/pile_19_token_sample.txt --save-inputs=./data2/sfft_pile18_$jobname.csv --outpath=\"./data2/pile18ft_$jobname.csv\" --log-path=\"./logs2/pile18ft_$jobname\" --model-name=\"$model\" --num-return-sequences=1 --collect-entropy" >> slurm.sh
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --no-threshold-knowledge --input-key="text" --batch-size=8 --dataset-path=./models/$jobname-ft/val.csv --save-inputs=./data2/val_$jobname.csv --outpath=\"./data2/val_$jobname.csv\" --log-path=\"./logs2/val_$jobname\" --model-name=\"$model\" --num-return-sequences=1" >> slurm.sh
    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --no-threshold-knowledge --input-key="query" --batch-size=8 --save-inputs=./data2/pft_on_sf_$jobname.csv --outpath=\"./data2/pft_on_sf_$jobname.csv\" --log-path=\"./logs2/pft_on_sf_$jobname\" --model-name=\"$model\" --num-return-sequences=1" >> slurm.sh

    # echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --save-inputs=./data2/sf_inputs_$jobname.csv --outpath=\"./data2/wikidata_$jobname.csv\" --log-path=\"./logs2/wikidata_$jobname\" --model-name=\"$model\" --num-return-sequences=10 --instruction=\"Accurately fill in the following sentence with the correct word, creating factual sentences as in the examples:\"" >> slurm.sh
    echo "TOKENIZERS_PARALLELISM=false python /home/mmahaut/projects/paramem/paramem/slot_filling.py test-generation --dataset-path=./benchmark/train.jsonl --input-key=context_query --save-inputs=./data2/sf_inputs_$jobname.csv --outpath=\"./data2/wikidata_$jobname.csv\" --log-path=\"./logs2/wikidata_$jobname\" --model-name=\"$model\" --num-return-sequences=1" >> slurm.sh
    # echo "python /home/mmahaut/projects/parametric_mem/slot_filling.py test-generation --outpath=\"./data2/wikidata_$jobname.csv\" --log-path=\"./logs2/wikidata_$jobname\" --model-name=\"$model\"" >> slurm.sh
    sed -i "12s/^/#SBATCH --job-name=sf_$jobname\n/" slurm.sh

    sbatch slurm.sh
    ((i=i+1))
done