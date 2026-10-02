# Paramem Workspace

This repository is organized for HPC/SLURM execution and experiment artifact hygiene.

## Execution policy

- Do not run Python directly on the login/root node.
- Use `sbatch` for batch jobs and `srun ... python ...` for Python execution.
- Environment bootstrap:
	- `source ~/.bashrc`
	- `module load CUDA/12.1.0` (when needed)
	- `conda activate paramem`

## Canonical layout

### Code

- `paramem/` — library/package code
- `scripts/analysis/` — analysis/eval/data-prep Python entrypoints
- `scripts/plots/` — plotting Python entrypoints
- `scripts/tests/` — Python test/validation entrypoints
- `scripts/entrypoints/` — canonical shell entrypoints (jobs/maintenance/tests)

### Documentation

- `docs/` — project guides, summaries, and technical notes
- `agent.md` — compact operational conventions
- `.github/copilot-instructions.md` — workspace policy for automation agents

### Results and logs

- `results/final/` — kept/canonical outputs
- `results/archive/` — superseded outputs and bundles
- `results/failed_runs/` — grouped failed logs
- `results/index/` — manifest and cleanup planning artifacts

## Entrypoints

Use canonical shell entrypoints from `scripts/entrypoints/...`.
Root-level `.sh` files are intentionally not used.

## Quick start

```bash
cd /home/mmahaut/projects/paramem

# Batch analysis jobs (wrapper paths retained)
sbatch scripts/entrypoints/jobs/run_analysis.sh
sbatch scripts/entrypoints/jobs/run_current_analysis.sh
sbatch scripts/entrypoints/jobs/run_layerwise_analysis.sh

# Memorization batch evaluation
./scripts/entrypoints/jobs/run_memorization_on_checkpoints.sh
./scripts/entrypoints/maintenance/check_memorization_status.sh
```
