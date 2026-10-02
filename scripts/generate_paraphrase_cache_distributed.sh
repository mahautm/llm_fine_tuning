#!/bin/bash
#SBATCH --job-name=gen_paraphrase_cache
#SBATCH --output=/home/mmahaut/projects/paramem/logs/paraphrase_gen_%A_%a.out
#SBATCH --error=/home/mmahaut/projects/paramem/logs/paraphrase_gen_%A_%a.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --array=0-7

# This is a SLURM job array - each task processes 1/8 of the questions
# With 8 workers, 200 questions = 25 questions per worker
# Each worker generates 25 × 5 = 125 paraphrases

# Load environment
module purge
module load cuda/11.8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate paramem

# Set paths
BASE_DIR="/home/mmahaut/projects/paramem"
WIKIDATA_PATH="${BASE_DIR}/data3/wikidata_Mis7.csv"
OUTPUT_DIR="${BASE_DIR}/data3/paraphrase_cache_parts"
FINAL_CACHE="${BASE_DIR}/data3/wikidata_paraphrase_llm_cache.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${BASE_DIR}/logs"

# Job array parameters
RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=${SLURM_ARRAY_TASK_COUNT}

echo "==================================================================="
echo "Paraphrase Cache Generation - Worker ${RANK}/${WORLD_SIZE}"
echo "==================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Check GPU availability
nvidia-smi

echo ""
echo "==================================================================="
echo "Generating Paraphrases"
echo "==================================================================="
echo ""

# Run paraphrase generation
cd "${BASE_DIR}"

python scripts/generate_paraphrase_cache.py \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --wikidata-path "${WIKIDATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-samples 200 \
    --num-paraphrases 5

echo ""
echo "==================================================================="
echo "Worker ${RANK} Complete"
echo "End time: $(date)"
echo "==================================================================="

# If this is the last worker (rank 7), merge all partial caches
if [ ${RANK} -eq $((WORLD_SIZE - 1)) ]; then
    echo ""
    echo "==================================================================="
    echo "Merging All Worker Caches"
    echo "==================================================================="
    
    # Wait a bit to ensure all workers have finished writing
    sleep 30
    
    python -c "
import json
import glob
import os

output_dir = '${OUTPUT_DIR}'
final_cache = '${FINAL_CACHE}'

print(f'Merging caches from: {output_dir}')

# Load all worker caches
merged_cache = {}
worker_files = sorted(glob.glob(os.path.join(output_dir, 'paraphrase_cache_worker_*.json')))

print(f'Found {len(worker_files)} worker cache files')

for worker_file in worker_files:
    print(f'Loading: {worker_file}')
    with open(worker_file, 'r') as f:
        worker_cache = json.load(f)
    merged_cache.update(worker_cache)
    print(f'  Added {len(worker_cache)} questions')

print(f'\\nTotal questions in merged cache: {len(merged_cache)}')

# Save merged cache
with open(final_cache, 'w') as f:
    json.dump(merged_cache, f, indent=2)

print(f'Merged cache saved to: {final_cache}')

# Print summary
print('\\n=== Paraphrase Cache Summary ===')
for i, (question, paraphrases) in enumerate(list(merged_cache.items())[:3]):
    print(f'\\nExample {i+1}: {question[:60]}...')
    for j, p in enumerate(paraphrases, 1):
        print(f'  {j}. {p[:60]}...')
"
    
    echo ""
    echo "==================================================================="
    echo "Merge Complete!"
    echo "Final cache: ${FINAL_CACHE}"
    echo "==================================================================="
fi
