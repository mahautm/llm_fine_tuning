# MP-Conditional Dataset Partitioning — Actionable Execution Plan

Last updated: 2026-03-25

## 1) Feasibility Audit: what exists vs what is missing

### Already available (reusable now)
- **Model + hidden state extraction**: `paramem/extract_hidden.py`, `paramem/extract_representations_utils.py` (supports HF + LoRA, per-layer states).
- **Cluster execution patterns**: multiple SLURM wrappers in `scripts/` and canonical guidance in `README.md` and `.github/copilot-instructions.md`.
- **Basic information-imbalance tooling**: `paramem/get_II.py` (currently monolithic, can be wrapped/refactored).
- **Core dependencies already in project**: `torch`, `transformers`, `accelerate`, `scikit-learn`, `dadapy`, plotting stack.

### Partially available (needs refactor for MP pipeline)
- **Activation handling**: extraction exists, but no standardized cache format for dataset-conditional MP runs.
- **Evaluation scripts**: several analysis scripts exist, but no dedicated Experiment-1 (MP partition stability) entrypoint.
- **Testing**: test scaffolding exists in repo, but no unit tests for spectral MP fitting utilities.

### Missing (must implement)
- `paramem/spectral/mp_fit.py` (MP fit, partition logic, diagnostics)
- `paramem/spectral/activation_cov.py` (stable covariance + basis projection)
- Dataset-conditional partition orchestration (`dataset_partition.py`)
- MP-focused CLI/runner script for first reproducible matrix over (dataset, layer)
- Synthetic validation tests (Wishart + spiked covariance)

---

## 2) Critical TODO clarifications / critiques

### A. Overly broad tasks need measurable completion criteria
Examples from current TODO:
- “Implement MP fitter” → **Done means**: synthetic Wishart recovers $\sigma^2$ and $\lambda_{\max}$ within tolerance; spikes are classified above bulk with high precision.
- “Activation caching works” → **Done means**: one command caches activations for 1 model × 1 dataset × selected layers and writes deterministic artifacts under `results/`.

### B. Separate “research” tasks from “engineering” tasks
Current list mixes paper framing with implementation. For momentum, split into:
1) **Engineering track** (this week): MP core + caching + Exp1 heatmap script
2) **Research track** (next): hypotheses validation, ablation protocol, writing

### C. Avoid ambiguous language around MP parameters
Current TODO sometimes assumes fixed $\gamma$. In practice, use effective
$$
\gamma_{\text{eff}} = \frac{\min(n,d)}{\max(n,d)}
$$
from observed activation matrix shape at each run.

### D. Add strict artifact conventions now
All new outputs should be under:
- `results/mp_reservoir/activations/...`
- `results/mp_reservoir/spectra/...`
- `results/mp_reservoir/figures/...`

This prevents scattering outputs across legacy folders.

---

## 3) Actionable execution backlog (priority ordered)

## P0 — Foundation (start now)
1. **Create spectral package**
   - Deliverables: `paramem/spectral/__init__.py`, `mp_fit.py`, `activation_cov.py`
   - Success checks: importable modules, deterministic synthetic tests.

2. **Implement MP fitting + partition API**
   - `fit_mp(eigenvalues, gamma=None)`
   - `fit_mp_mle(eigenvalues, gamma_grid=None)`
   - `partition(eigenvalues, lambda_max, atol=...)`
   - `generate_mp_pdf_grid(...)`

3. **Implement activation covariance utilities**
   - `compute_cov(activations, center=True, shrinkage=None)`
   - `compute_eigenspectrum(cov)`
   - `project_onto_singular_basis(cov, right_singular_vectors)`

4. **Add synthetic tests**
   - Wishart recovery and spike separation.

## P1 — First end-to-end experiment (Exp1 seed)
5. **Create first runner script**
   - `scripts/analysis/run_mp_partition_stability.py`
   - Inputs: activation cache or pickles, dataset names, layers
   - Outputs: per-(dataset, layer) `lambda_max`, signal fraction csv in `results/`.

6. **SLURM launcher for MP Exp1**
   - `scripts/entrypoints/jobs/run_mp_partition_stability.sh`
   - Must follow environment policy (`source ~/.bashrc`, optional CUDA module, `conda activate paramem`, then `srun ... python ...`).

## P2 — Next immediately after Exp1 baseline
7. **Dataset partition orchestrator** (`dataset_partition.py`)
8. **Refactor II code to `paramem/probing/dii.py` with clean API**
9. **Polysemantic decomposition utilities (Exp2 prep)**

---

## 4) This-week practical schedule (tight loop)

- **Day 1**: P0-1/2/4 complete (MP core + tests)
- **Day 2**: P0-3 and P1-5 complete (covariance + first runner)
- **Day 3**: P1-6 + first cluster submission for 1 model × 1 dataset × 1 layer
- **Day 4-5**: expand to full Exp1 matrix and first heatmap

---

## 7) Execution policy update (effective now)

- Do not run MP experiments on tiny models (e.g., 70M) except for pure smoke tests.
- Minimum experimental model size is **>=1B parameters**.
- Prefer Pythia scale points: `EleutherAI/pythia-1b`, `EleutherAI/pythia-1.4b` for early Exp1.
- Run activation caching via **SLURM job arrays** across (model, dataset) combinations.
- Run partition aggregation as a dependent array stage after cache completion.

Reference scripts:
- `scripts/entrypoints/jobs/mp_activation_array_tasks.tsv`
- `scripts/entrypoints/jobs/run_mp_activation_cache_array.sh`
- `scripts/entrypoints/jobs/run_mp_partition_by_model_array.sh`
- `scripts/entrypoints/jobs/submit_mp_array_pipeline.sh`

---

## 5) Immediate commands to run (after code lands)

Use cluster-safe invocation patterns:
- `sbatch scripts/entrypoints/jobs/run_mp_partition_stability.sh`
- or interactive test: `srun --gres=gpu:1 --partition=alien --qos=alien python -m scripts.analysis.run_mp_partition_stability ...`

---

## 6) Definition of done for “started MP approach”

The approach is considered truly started when all are true:
1. MP fitting utilities merged and tested.
2. One reproducible Exp1 runner produces `lambda_max` + signal fraction outputs in `results/mp_reservoir/`.
3. One SLURM wrapper exists and can launch that runner without manual patching.
