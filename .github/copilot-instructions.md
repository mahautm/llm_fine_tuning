# Copilot Workspace Instructions

## Execution Environment (HPC / SLURM)

- Do not run `python` directly on the login/root node.
- Use SLURM execution paths for Python workloads:
  - `sbatch ...` for batch jobs
  - `srun ... python ...` for interactive or in-job task launch
- Ensure environment setup before Python execution:
  - `source ~/.bashrc`
  - `module load CUDA/12.1.0` (when required)
  - `conda activate paramem`
- common example:
```srun --partition=alien --qos=alien --gres=gpu:1 --time=00:20:00 bash -lc 'source ~/.bashrc && conda activate paramem && python -m [SCRIPT]'
```
- only use `node044` for quick interactive tests, not for training or long-running jobs. For long-running jobs, use --exclude=node044 in sbatch/srun.

## Repository Conventions

- Generated experiment artifacts should live under `results/`.
- Keep source scripts in place unless a refactor explicitly updates all references.
- Prefer non-destructive cleanup first (index/plan), then execution.