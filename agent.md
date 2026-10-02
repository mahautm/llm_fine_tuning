# Agent Operating Guide

## Status: CLEANUP COMPLETE ✅

**Phase ABCD cleanup executed 2026-03-25.**
- Root directory cleaned of 313 MB clutter (all `.err`, `.out`, `.zip`, `plots_*`)
- 96 plot files + 46 log files reorganized into canonical `results/` tree
- 2 canonical successful runs tagged: `run_id=2033704` (full-FT) and `run_id=2033703` (LoRA)
- See [results/index/CLEANUP_COMPLETION.md](results/index/CLEANUP_COMPLETION.md) for full details

## Repository Organization (Canonical Layout - NOW LIVE)

Use this layout for all new analysis outputs, logs, and archived artifacts:

```
results/
  index/
    run_manifest.csv
    failure_signatures.csv
  final/
    layerwise/
    probing/
    memorization/
  archive/
    plots_zips/
    old_plots/
    old_logs/
  failed_runs/
    root_logs/
    slurm_logs/

scripts/
  analysis/   # analysis/evaluation/data-prep Python entrypoints
  plots/      # plotting Python entrypoints
  tests/      # Python validation/test scripts

docs/         # project guides, summaries, and technical notes
```

## Entrypoints

- Canonical shell entrypoints live under `scripts/entrypoints/{jobs,maintenance,tests}/`.
- Root-level `.sh` files have been removed.
- Use explicit paths in docs/automation (for example, `scripts/entrypoints/jobs/run_analysis.sh`).

## Code and docs layout

```
scripts/
  analysis/   # analysis/evaluation/data-prep Python entrypoints
  plots/      # plotting Python entrypoints
  tests/      # Python validation/test scripts

docs/         # project guides, summaries, and technical notes
```

## What Goes Where

- `results/final/`
  - Final accepted plots and CSV metrics for reporting.
  - Keep the most recent successful run per family + dataset + mode.

- `results/archive/`
  - Superseded plots, zip bundles, and older logs retained for traceability.
  - Archive first; do not hard-delete immediately.

- `results/failed_runs/`
  - Logs from failed runs, grouped by source (`root_logs`, `slurm_logs`).
  - Keep compact failure evidence (error signature + tail excerpts) for recurring issue tracking.

- `results/index/`
  - `run_manifest.csv` as the source of truth for run status and provenance.
  - `failure_signatures.csv` for categorized failures (`OOM`, `TIME_LIMIT`, `ENV`, `PATH`, `NCCL`, `TRACEBACK`).

## Mandatory Conventions

1. New scripts must default output to `results/final/<family>/` unless explicitly overridden.
2. No new top-level `plots_*` directories.
3. No new loose root-level `.err`/`.out` files.
4. Every kept artifact must be traceable to a run in `results/index/run_manifest.csv`.
5. Before cleanup deletions: create archive copy + update manifest first.

## Migration Notes (Current Repo)

- Existing root loose logs (`analysis_*.err`, `layerwise_*.err`, `current_analysis_*.err`, `nccl_test_*.err`) were moved to `results/failed_runs/root_logs/`.
- Existing `plots_8b_layerwise*.zip` and `graphs.zip` were moved to `results/archive/plots_zips/`.
- Existing `plots_*` png outputs were normalized under `results/final/plots/<family>/` after manifest tagging.

## Reference

- Strategy source: `CLEANUP_STRATEGY.md`
- Execution policy source: `.github/copilot-instructions.md`
