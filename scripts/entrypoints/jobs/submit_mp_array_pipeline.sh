#!/bin/bash
set -euo pipefail

cd /home/mmahaut/projects/paramem

CACHE_JOB=$(sbatch --parsable scripts/entrypoints/jobs/run_mp_activation_cache_array.sh)
PARTITION_JOB=$(sbatch --parsable --dependency=afterok:${CACHE_JOB} scripts/entrypoints/jobs/run_mp_partition_by_model_array.sh)

echo "Submitted cache array job: ${CACHE_JOB}"
echo "Submitted partition array job (afterok): ${PARTITION_JOB}"
