# Exclude node044 and node043 from Slurm jobs
EXCLUDE_NODES="node044,node043"
CHECKPOINT_PATH="$1"
USE_LORA="$2"
OUT_DIR="${3:-${CHECKPOINT_PATH}/slurm_logs}"

# Set variables
PERF_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/performance_evaluation.py"
ID_SCRIPT="/home/mmahaut/projects/paramem/paramem/evaluation/ID_matrixent_evaluation.py"
mkdir -p "$OUT_DIR"

PERF_OUT="${OUT_DIR}/performance_evaluation.out"
PERF_ERR="${OUT_DIR}/performance_evaluation.err"
ID_OUT="${OUT_DIR}/ID_matrixent.out"
ID_ERR="${OUT_DIR}/ID_matrixent.err"

# # Launch performance_evaluation.py job with job name
PERF_JOBID=$(sbatch --job-name=perf_eval --gres=gpu:2 --mem=200G --partition=alien --qos=alien \
    --exclude=$EXCLUDE_NODES \
    --output="$PERF_OUT" --error="$PERF_ERR" \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA \
    --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd paramem && poetry run python $PERF_SCRIPT" | awk '{print $4}')

# Launch ID_matrixent.py job with job name
ID_JOBID=$(sbatch --job-name=id_matrixent --gres=gpu:2 --mem=200G --partition=alien --qos=alien \
    --exclude=$EXCLUDE_NODES \
    --output="$ID_OUT" --error="$ID_ERR" \
    --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,USE_LORA=$USE_LORA \
    --wrap="source ~/.bashrc && module load CUDA/12.1.0 && conda activate paramem && cd paramem && CHECKPOINT_PATH=$CHECKPOINT_PATH USE_LORA=$USE_LORA poetry run python $ID_SCRIPT" | awk '{print $4}')
echo "ID_JOBID: $ID_JOBID"