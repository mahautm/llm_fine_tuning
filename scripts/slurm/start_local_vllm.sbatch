#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/vllm_server_%j.out

source ~/.bashrc
conda activate paramem
module load CUDA/12.1.0

# Select the model you want to deploy locally:
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
PORT=8000

# Get the host node IP
HOST_IP=$(hostname -i)

echo "Starting vLLM OpenAI-compatible server on $HOST_IP:$PORT"
echo "Model: $MODEL_ID"
echo "To use this, set your script's --local_api_base=http://$HOST_IP:$PORT/v1"

# Check if vLLM is installed in the current environment, install if missing
if ! python -c "import vllm" &> /dev/null; then
    echo "vLLM is not installed in the current environment. Installing now..."
    export CUDA_HOME=$CUDA_ROOT
    pip install numpy "vllm==0.4.0" --extra-index-url https://download.pytorch.org/whl/cu121
else
    echo "vLLM is already installed."
fi

# Launch vLLM OpenAI API server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port $PORT \
    --served-model-name $MODEL_ID
