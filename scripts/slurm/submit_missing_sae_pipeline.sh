#!/bin/bash
set -euo pipefail

cd /home/mmahaut/projects/paramem

# H2 is missing; launch architecture array.
H2_JOB=$(sbatch --parsable scripts/slurm/H2_train_sae_array.sbatch)

# H3 depends on all H2 array tasks.
H3_JOB=$(sbatch --parsable --dependency=afterok:${H2_JOB} scripts/slurm/H3_eval_pareto_mp.sbatch)

# H4 and H5 previously failed; relaunch them as dependent stages after H3.
H4_JOB=$(sbatch --parsable --dependency=afterok:${H3_JOB} scripts/slurm/H4_auto_annotation.sbatch)
H5_JOB=$(sbatch --parsable --dependency=afterok:${H3_JOB} scripts/slurm/H5_causal_intervention.sbatch)

echo "Submitted H2 array job: ${H2_JOB}"
echo "Submitted H3 job (afterok H2): ${H3_JOB}"
echo "Submitted H4 job (afterok H3): ${H4_JOB}"
echo "Submitted H5 job (afterok H3): ${H5_JOB}"
