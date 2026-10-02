# Paramem Results Audit + Cleanup Strategy

Date: 2026-03-24

## Execution status (updated 2026-03-25)

Cleanup phases A/B/C were executed, followed by a re-audit and an extra root-artifact pass.

- Root loose artifacts now at zero:
  - `*.err`: 0
  - `*.out`: 0
  - `*.zip`: 0
  - `*.png`: 0
  - `*.log`: 0
  - `plots_*` dirs: 0
- Reorganized outputs currently in `results/`:
  - `results/final`: 98 files
  - `results/archive`: 27 files
  - `results/failed_runs`: 37 files
  - `results/index`: 6 files
- Additional post-Phase-D consolidation done:
  - 131 root-generated `.png` moved to `results/archive/old_plots/root_generated/`
  - 2 root `.log` files moved to `results/archive/old_logs/root_generated/`

Primary completion artifacts:
- `results/index/run_manifest.csv`
- `results/index/failure_signatures.csv`
- `results/index/move_plan.csv`
- `results/index/phase_c_artifact_plan.csv`
- `results/index/CLEANUP_COMPLETION.md`

Note: The audit snapshot below captures the **pre-cleanup** state used to define the strategy.

## 1) Current State Audit (evidence-based snapshot)

### Storage hotspots (top-level)
- `benchmark/`: ~42G (dominant footprint)
- `data/`: ~1.7G
- `wandb/`: ~1.1G
- `logs/`: ~438M
- `data2/`: ~245M
- `logs2/`: ~131M
- `data3/`: ~128M
- `output/`: ~29M
- `slurm_logs/`: ~11M

### Result artifact structure currently present
- Layerwise outputs exist under `output/layerwise_comparison`, `output/layerwise_probing`, `output/layerwise_probing_8b`, and `output/layerwise_self_comparison`.
- Plot outputs are spread across multiple top-level dirs:
  - `plots_8b_layerwise` (41 files)
  - `plots_8b_layerwise_models3` (61 files)
  - `plots_8b_probing` (20 files)
  - `plots_8b_lora` (5 files)
  - `plots_memorization` (10 files)
- Multiple zip bundles for similar plot families are present (`plots_8b_layerwise*.zip`, `graphs.zip`), suggesting version drift and duplication.

### Log/error health snapshot
- `slurm_logs`: 40 `.err` + 40 `.out` files (80 total files)
- `logs`: 107 files
- Root-level `.err` and `.out`: 11 each

From pattern scan of `slurm_logs/*.err`:
- ~23/40 files contain traceback/error signatures.
- OOM appears in at least one recent full-FT resume run (`oom-kill` in cgroup).
- Time-limit cancellation appears in at least one full-FT run.
- 7 runs show `wandb: Synced`/`View run`, indicating likely successful completions.

Root-level failures include:
- stale/mismatched script paths (`can't open file ... analyze_lora_vs_full_ft.py`, `create_layerwise_checkpoint_plots.py`)
- environment launch errors in NCCL tests (`torchrun: No such file or directory`, `bash: No such file or directory`)

## 2) Hypotheses (best-effort, based on artifacts)

These are informed guesses, not definitive conclusions:

1. **LoRA is the most operationally stable training mode in this repo state**
   - Multiple summaries and logs indicate successful LoRA runs with synced W&B artifacts.
   - Full FT appears more frequently exposed to OOM/time-limit failure modes.

2. **Many failures were infrastructure/environmental rather than scientific**
   - Missing script path issues and module/conda/cuda setup errors recur.
   - NCCL test errors indicate launcher/runtime path drift across nodes.

3. **Result interpretation is currently slowed more by artifact sprawl than by missing compute**
   - Outputs are split across multiple similarly named dirs and zip snapshots.
   - There is no single canonical “latest accepted” result index.

4. **Layerwise/probing pipelines likely produced useful data, but provenance is fragmented**
   - Structured outputs exist in `output/` with multiple subfamilies.
   - Summary docs describe methodology, but success/failure status per run is not centralized.

## 3) Cleanup Strategy (safe, phased)

## Phase A — Freeze and index before deleting (no-risk)

1. Create canonical structure:
   - `results/final/`
   - `results/archive/`
   - `results/failed_runs/`
   - `results/index/`

2. Build a run manifest (`results/index/run_manifest.csv`) with columns:
   - `run_id, family, mode(full|lora|analysis), dataset, status(success|failed|unknown), reason, has_wandb, output_path, log_err, log_out, kept_as`

3. Define “success” operationally (for indexing only):
   - contains `wandb: Synced` OR explicit completion marker in `.out`
   - no terminal traceback/oom/time-limit marker in tail section

## Phase B — Canonicalize retained artifacts

Keep only one canonical destination per experiment family:

- **Keep (final)**
  - most recent successful run per family+dataset+mode
  - plots directly referenced by docs/reports
  - final CSV metrics used to generate kept plots

- **Archive (compressed, not deleted immediately)**
  - older superseded plots/zip bundles
  - intermediate checkpoints/outputs not referenced in final analysis
  - old slurm logs for successful runs once manifest is complete

- **Failed runs bucket**
  - keep only `.err` tail excerpts + job metadata for repeated failure classes
  - delete/recompress full verbose logs after extracting signatures

## Phase C — Remove top-level clutter

After manifest validation:

1. Move root loose logs (`analysis_*.err`, `layerwise_*.err`, `current_analysis_*.err`, `nccl_test_*.err`) into `results/failed_runs/root_logs/`.
2. Move plot zips into `results/archive/plots_zips/` and retain only one preferred zip per family/date.
3. Move standalone plot png families into `results/final/plots/<family>/` with normalized naming.
4. Keep `output/` as source-of-truth for regenerated plots; avoid parallel “plots_*” drift long-term.

## Phase D — Retention policy (ongoing)

### Logs
- Keep full `.err/.out` for last N=10 successful runs per family.
- Keep full logs for all runs in last 30 days.
- For older failed runs, retain only:
  - first 60 lines
  - last 120 lines
  - extracted signature (`OOM`, `TIME_LIMIT`, `ENV`, `PATH`, `NCCL`, `TRACEBACK`)

### Artifacts
- Keep one canonical final plot per metric variant.
- Keep raw CSVs that generated those plots.
- Archive redundant plot versions monthly.

### Data
- `benchmark/` is large but likely core input; do not delete blindly.
- If duplicated subsets exist across `data/`, `data2/`, `data3/`, deduplicate by checksum and symlink/index.

## 4) Suggested directory target layout

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
```

## 5) Minimal execution checklist

1. Build manifest from existing `slurm_logs/*.err` + `*.out` + root `.err/.out`.
2. Mark run status (`success/failed/unknown`) with explicit evidence string.
3. Tag one canonical run per family.
4. Move files to `results/{final,archive,failed_runs}`.
5. Regenerate all “kept” plots from canonical CSVs once, verify reproducibility.
6. Delete only after one backup snapshot and manifest review.

## 6) High-value quick wins

1. **Immediate**: move root loose `.err/.out` into one folder.
2. **Immediate**: consolidate `plots_8b_layerwise*.zip` into one archive location.
3. **Immediate**: create `results/index/run_manifest.csv` to stop further sprawl.
4. **Next**: enforce one output path per script via CLI default (`--output-dir results/final/...`).

---

If you want, the next step is to automatically generate the first `run_manifest.csv` from current logs and move files into the proposed tree without deleting anything.