#!/bin/bash
#SBATCH --job-name=ft4_full_v3_resume
#SBATCH --output=/home/mmahaut/projects/paramem/slurm_logs/ft4_full_v3_resume_%j.out
#SBATCH --error=/home/mmahaut/projects/paramem/slurm_logs/ft4_full_v3_resume_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:4
#SBATCH --mem=250G
#SBATCH --time=144:00:00
#SBATCH --exclude=node044,node041

source ~/.bashrc
conda activate paramem
cd /home/mmahaut/projects/paramem

module load CUDA/12.1.0
export CUDA_HOME=$CUDA_HOME

export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"
# Optional: free space automatically after eval
# export KEEP_CHECKPOINT=0

torchrun --nproc_per_node=4 --master_port=29511 \
  /home/mmahaut/projects/paramem/paramem/fine_tune_with_ckpts.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --dataset-name "/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv" \
  --output-dir "/home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4-v3" \
  --per-device-train-batch-size 1 \
  --num-train-epochs 5 \
  --save-steps 130 \
  --logging-steps 10 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --block-size 166 \
  --deep-speed \
  --resume-from-checkpoint "/home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4-v3/checkpoint-780" \
  --eval-script "/home/mmahaut/projects/paramem/scripts/launch_memorization_checkpoint_eval.sh"
