#!/bin/bash
set -euo pipefail

cd /home/mmahaut/projects/paramem

H1_JOB=$(sbatch --parsable scripts/slurm/H1_extract_rich_domains.sbatch)
H2_JOB=$(sbatch --parsable --dependency=afterok:${H1_JOB} scripts/slurm/H2_train_sae_array.sbatch)
H3_JOB=$(sbatch --parsable --dependency=afterok:${H2_JOB} scripts/slurm/H3_eval_pareto_mp.sbatch)
H4_JOB=$(sbatch --parsable --dependency=afterok:${H3_JOB} scripts/slurm/H4_auto_annotation.sbatch)

echo "Submitted H1 rich extraction job: ${H1_JOB}"
echo "Submitted H2 array job (afterok H1): ${H2_JOB}"
echo "Submitted H3 job (afterok H2): ${H3_JOB}"
echo "Submitted H4 job (afterok H3): ${H4_JOB}"
