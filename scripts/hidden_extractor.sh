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
models=("meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-v0.3")
ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model.bin" "./models/Met7i-ft/checkpoint-4000/pytorch_model.bin" "./models/Mis7-ft/checkpoint-2000/pytorch_model.bin")
# ckpts=("./models/Met7-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/Met7i-ft/checkpoint-4000/pytorch_model_fsdp_0" "./models/Mis7-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/Mis7i-ft/checkpoint-2000/pytorch_model_fsdp_0" "./models/OLM7-ft/checkpoint-2000/pytorch_model_fsdp_0")
# ckpts=("./models/Met7-pileft/checkpoint-6000/pytorch_model_fsdp_0" "./models/Met7i-pileft/checkpoint-6000/pytorch_model_fsdp_0" "./models/Mis7-pileft/checkpoint-6000/pytorch_model_fsdp_0")
# models=("allenai/OLMo-1.7-7B-hf")
# ckpts=("./models/OLM7-ft/checkpoint-2000/pytorch_model_fsdp_0")

execute_slurm() {
    sed -i "12s/^/#SBATCH --job-name=hlft-$jobname\n/" slurm.sh
    sbatch slurm.sh
}

i=0
for model in "${models[@]}"
do
#     # Set jobname variable
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
    data_path="./data/wikidata_$jobname.csv"
#     # reset variable to be PILE
    # jobname="$jobname"P
    # base
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=query --out-pickle-prefix=\"./hlayer2/$jobname-base\"" >> slurm.sh
    # execute_slurm
    # ft
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --checkpoint=\"${ckpts[$i]}\" --input-key=query --out-pickle-prefix=\"./hlayer2/$jobname-ft-small\"" >> slurm.sh
    # execute_slurm
    # icl
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context_query --out-pickle-prefix=\"./hlayer2/$jobname-icl\"" >> slurm.sh
    # execute_slurm
    # sanity-check 1
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context1_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity1\" --sanity-check" >> slurm.sh
    # execute_slurm
    # sanity-check 2
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context2_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity2\" --sanity-check" >> slurm.sh
    # execute_slurm
    # sanity-check 3
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context1_query2 --out-pickle-prefix=\"./hlayer2/$jobname-sanity3\" --sanity-check" >> slurm.sh
    # execute_slurm
    # sanity-check 1 (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=context1_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity1-ft\" --sanity-check" >> slurm.sh
    execute_slurm
    # sanity-check 2 (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=context2_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity2-ft\" --sanity-check" >> slurm.sh
    execute_slurm
    # sanity-check 3 (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=context1_query2 --out-pickle-prefix=\"./hlayer2/$jobname-sanity3-ft\" --sanity-check" >> slurm.sh
    execute_slurm
    # sanity-check 1 -- knowns
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context1_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity1-kn\" --sanity-check --kn-threshold=0.9" >> slurm.sh
    # execute_slurm
    # sanity-check 2 -- knowns
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context2_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity2-kn\" --sanity-check --kn-threshold=0.9" >> slurm.sh
    # execute_slurm
    # sanity-check 3 -- knowns
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=context1_query2 --out-pickle-prefix=\"./hlayer2/$jobname-sanity3-kn\" --sanity-check --kn-threshold=0.9" >> slurm.sh
    # execute_slurm
    # sanity-check 1 -- knowns (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=context1_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity1-ft-kn\" --sanity-check --kn-threshold=0.9" >> slurm.sh
    execute_slurm
    # sanity-check 2 -- knowns (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=context2_query1 --out-pickle-prefix=\"./hlayer2/$jobname-sanity2-ft-kn\" --sanity-check --kn-threshold=0.9" >> slurm.sh
    execute_slurm
    # sanity-check 3 -- knowns (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=context1_query2 --out-pickle-prefix=\"./hlayer2/$jobname-sanity3-ft-kn\" --sanity-check --kn-threshold=0.9" >> slurm.sh
    execute_slurm
    # icl -- knowns (ft)
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --checkpoint-path=\"${ckpts[$i]}\" --input-key=answer --out-pickle-prefix=\"./hlayer2/$jobname-sanity-icl-ft\" --sanity-check" >> slurm.sh
    # execute_slurm
    # icl -- knowns
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=answer --out-pickle-prefix=\"./hlayer2/$jobname-sanity-icl\" --sanity-check" >> slurm.sh
    # execute_slurm
    # no context
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --input-key=query --out-pickle-prefix=\"./hlayer2/$jobname-nc\" --no-context" >> slurm.sh
    # execute_slurm
    # no context (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $data_path --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --input-key=query --out-pickle-prefix=\"./hlayer2/$jobname-nc-ft\" --no-context" >> slurm.sh
    execute_slurm
    pile_data_9=/home/mmahaut/projects/paramem/data/pile_9_token_sample.txt
    # pile 9 data
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $pile_data_9 --out-pickle-prefix=\"./hlayer2/$jobname-pile8\"" >> slurm.sh
    # execute_slurm
    # pile 9 data (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $pile_data_9 --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --out-pickle-prefix=\"./hlayer2/$jobname-pile8-ft\"" >> slurm.sh
    execute_slurm
    pile_data_19=/home/mmahaut/projects/paramem/data/pile_19_token_sample.txt
    # pile 19 data
    # echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $pile_data_19 --out-pickle-prefix=\"./hlayer2/$jobname-pile18\"" >> slurm.sh
    # execute_slurm
    # pile 19 data (ft)
    echo "python /home/mmahaut/projects/paramem/paramem/extract_hidden.py \"$model\" 10 $pile_data_19 --override-data-file --checkpoint-path=\"${ckpts[$i]}\" --out-pickle-prefix=\"./hlayer2/$jobname-pile18-ft\"" >> slurm.sh
    execute_slurm
    ((i=i+1))
done
