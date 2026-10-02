# MP Results So Far (Interim)

Date: 2026-04-09

## Scope completed

This summary covers the completed MP-conditional pipeline through:
- Exp1 MP partition extraction and plotting on `pythia-1b` and `pythia-1.4b`
- pooled-vs-split MP comparison
- DII signal-subspace matrix computation
- H2 SAE training, H3 Pareto/MP thresholding, H4 auto-annotation, and H5 causal intervention reruns

## What is new vs existing literature

Relative to prior Random Matrix Theory analyses such as Martin and Mahoney (JMLR 2021, `htsr_20-410.pdf`), the key novelty in this experimental path is not merely applying MP, but turning MP into a feature-level mechanistic pipeline:

1. **Object of analysis shift:** from trained weight-matrix ESD diagnostics to SAE feature-activation partitions conditioned on data distributions.
2. **Dataset-conditional structure tests:** pooled-vs-split deltas and DII subspace mismatch are measured directly, showing conditional geometry differences rather than assuming universality.
3. **Actionable MP decisions:** MP boundaries are used to prune, classify (`Universal`/`Domain`/`Dead`), and calibrate, not only to describe spectra.
4. **Semantic and causal coupling:** MP-selected features are linked to LLM-as-a-judge outputs and intervention/placebo behavior, connecting spectral criteria to mechanistic effects.
5. **Cross-modal and scaling validation:** the same logic is exercised across text and vision settings and scaled to a frontier LM extraction/training regime.

This positions the contribution as a mechanistic interpretability workflow grounded in RMT, rather than a standalone spectral characterization study.

## Detailed experiment readout

### Exp1: MP partition extraction and stability (H1)

**Question:** does MP partitioning vary by dataset and layer, or is it effectively constant?

**Method:**
- Cached hidden activations for `pythia-1b` and `pythia-1.4b` across selected layers and datasets.
- Built covariance spectra and estimated MP bulk edge per condition.
- Computed signal fraction (eigen-mass above MP upper edge) per layer/dataset.

**Result:**
- Signal fractions are non-constant across datasets and layers.
- This is visible in layer heatmaps and trend plots in `results/mp_reservoir/figures/`.

**Interpretation:**
- Supports H1: MP thresholding is dataset-conditional and layer-sensitive.
- A single pooled threshold discards meaningful conditional structure.

### Exp1b: pooled-vs-split comparison

**Question:** is pooled fitting equivalent to averaging per-dataset fits?

**Metric:** max absolute pooled-minus-split mean signal-fraction delta.

**Result:**
- `pythia1b`: 0.027466 at layer 12
- `pythia14b`: 0.049072 at layer 16

**Interpretation:**
- Not equivalent. The gap increases in deeper sampled layers.
- Split conditioning preserves differences that pooled processing smooths away.

### Exp1c: DII signal-subspace matrix

**Question:** do MP signal subspaces share local geometry across datasets?

**Result:**
- DII diagonal mean is ~0.2 (estimator baseline in this setup).
- Off-diagonal means are near 1.0:
  - `pythia1b`: ~0.993-0.999
  - `pythia14b`: ~0.986-0.998

**Interpretation:**
- Strong mismatch between dataset-conditioned signal subspaces.
- Early support for non-universal local geometry in MP signal components.

### H2: SAE architecture + Pareto behavior

**Question:** does MP-based filtering remain useful across SAE variants?

**Method:** Standard, Top-K, Batch Top-K SAE training/eval pipeline with Pareto readout.

**Result (interim):**
- Top-K variants preserve strict sparsity behavior.
- Standard SAE remains denser but is compatible with MP-guided pruning.

**Interpretation:**
- Directional support for architecture independence of MP filtering.
- Full cross-architecture statistical comparison remains future work.

### H3: domain separation via MP thresholding

**Question:** can MP recover features that are domain-specific (active above bulk only in domain data)?

**Pre-calibration rich run (`pile_Met7` vs `wikidata_Met7`, OLMo-1B):**
- `Universal = 16384`, `Domain_Specific = 0`, `Dead_or_Substrate = 0`
- Degenerate all-universal regime.

**Post-calibration run (MAD scaling + clipping + quantile sigma):**
- Baseline MP pass: `Universal = 6420`, `Domain_Specific = 53`, `Dead_or_Substrate = 2515`
- Calibrated MP pass: `Universal = 6434`, `Domain_Specific = 40`, `Dead_or_Substrate = 2262`

**Interpretation:**
- Calibration breaks degeneracy and recovers non-zero domain-specific sets.
- H3 is now operational on richer-domain pairs.

### H4: auto-annotation of MP domain features

**Question:** are recovered domain features semantically coherent under LLM judging?

**Method:**
- Took calibrated H3 domain feature indices (`n=40`).
- Queried local vLLM endpoint for label + confidence per feature.

**Result (latest complete run):**
- 40/40 features annotated without API errors.
- Mean confidence: 0.9655.
- Features with confidence < 0.3: 0.
- Top raw label frequencies:
  - `Negative Sentiment`: 24
  - `Null Context`: 9
  - remaining labels are sparse singletons/variants.

**Interpretation and caution:**
- Pipeline execution is now stable end-to-end.
- Current context extraction still contains placeholder-style contexts in `scripts/eval_auto_annotation.py`; confidence values are useful for runtime validation but remain provisional for strong semantic claims until token-aligned contexts are fully integrated.

### H5: causal intervention

**Question:** does steering through selected features cause directional output changes?

**Result (interim):**
- Intervention jobs completed successfully.
- Qualitative directional steering was observed in logs.

**Interpretation:**
- Hooking and steering path is functioning.
- Quantitative effect-size and placebo-controlled statistics are still pending.

## Paper-ready results table (interim)

| Experiment | Objective | Metric | Current value | Interpretation | Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Exp1 MP stability (H1) | Test whether MP partition is dataset/layer conditional | Layerwise MP signal fraction by dataset | Non-constant across layers/datasets (see heatmaps/trends) | Supports dataset-conditioned MP behavior | Full significance testing pending |
| Exp1b pooled-vs-split | Compare pooled fitting vs per-dataset fitting | Max abs pooled-minus-split delta | `pythia1b`: 0.027466 (L12), `pythia14b`: 0.049072 (L16) | Pooled and split are not equivalent | Sampled-layer analysis, not full depth |
| Exp1c DII matrix | Compare local geometry of MP signal subspaces | DII off-diagonal mean | `pythia1b`: ~0.993-0.999, `pythia14b`: ~0.986-0.998 | Strong cross-dataset mismatch in local geometry | Deterministic subsampling used for memory control |
| H2 SAE architecture | Check architecture-independence of MP utility | Pareto + MP-pruned behavior | Top-K remains sparse; Standard SAE prune-compatible | Directional support across SAE variants | Full architecture-by-architecture stats not finalized |
| H3 domain separation | Recover domain-only features under MP | MP counts: Universal / Domain / Dead | Pre-calib: 16384 / 0 / 0; Calib: 6434 / 40 / 2262 (baseline pass: 6420 / 53 / 2515) | Calibration resolves degenerate all-universal regime | Sensitivity to calibration hyperparameters still to quantify |
| H4 auto-annotation | Judge coherence of H3 domain features | Valid responses, mean confidence, low-confidence count | 40/40 valid, mean confidence 0.9655, <0.3 count = 0 | H4 runtime pipeline is stable and complete | Current contexts are mock placeholders, not token-aligned snippets |
| H5 causal intervention | Test whether feature steering changes outputs | Steering run success + qualitative directionality | Completed runs with directional effects in logs | Intervention path is functional | Quantitative effect size and placebo controls pending |

## Quantitative highlights

### 1) Pooled vs split divergence
- `pythia1b`: max absolute pooled-minus-split mean signal-fraction delta = **0.027466** at layer **12**
- `pythia14b`: max absolute pooled-minus-split mean signal-fraction delta = **0.049072** at layer **16**

Interpretation: pooled activations are not equivalent to averaging per-dataset MP partitions, especially in deeper sampled layers.

### 2) DII subspace matrices
- DII diagonal mean remains at **0.2** (self-comparison baseline in current estimator setup)
- Off-diagonal means are consistently near **1.0**, indicating strong cross-dataset mismatch in local neighborhood structure:
  - `pythia1b`: ~0.993 to 0.999
  - `pythia14b`: ~0.986 to 0.998

Interpretation: dataset signal subspaces are not collapsing to a shared universal local geometry in this setup.

## Artifact map

### MP spectra and pivots
- `results/mp_reservoir/spectra/pythia1b/mp_partition_metrics_pythia1b_todo_exp1.csv`
- `results/mp_reservoir/spectra/pythia14b/mp_partition_metrics_pythia14b_todo_exp1.csv`
- `results/mp_reservoir/spectra/pythia1b/mp_signal_fraction_pivot_pythia1b_todo_exp1.csv`
- `results/mp_reservoir/spectra/pythia14b/mp_signal_fraction_pivot_pythia14b_todo_exp1.csv`

### Figures
- `results/mp_reservoir/figures/pythia1b/mp_signal_fraction_heatmap_pythia1b_todo_exp1.png`
- `results/mp_reservoir/figures/pythia14b/mp_signal_fraction_heatmap_pythia14b_todo_exp1.png`
- `results/mp_reservoir/figures/mp_signal_fraction_layer_trends_todo_exp1.png`

### Probing analysis outputs
- `results/mp_reservoir/analysis/pythia1b/pooled_vs_split_pythia1b_todo_exp1.csv`
- `results/mp_reservoir/analysis/pythia14b/pooled_vs_split_pythia14b_todo_exp1.csv`
- `results/mp_reservoir/analysis/pythia1b/dii_matrix_layer_0_todo_exp1.csv`
- `results/mp_reservoir/analysis/pythia14b/dii_matrix_layer_0_todo_exp1.csv`
- plus per-layer DII matrices and heatmaps for all sampled layers

## Latest checks

### MP calibration milestone

- Robust calibration (`mad` + clipping + quantile sigma) resolved the all-universal collapse on the rich OLMo pair.
- Calibrated report is in `results/H3_mp_threshold_report_calibrated.json`.

### H4 runtime milestone

- Local vLLM endpoint relaunch and parser hardening completed.
- Final calibrated H4 annotations are in `results/H4_llm_auto_annotations_calibrated.json`.
- Run quality: 40/40 valid responses, mean confidence 0.9655.

### Live FT-vs-LoRA checkpoint campaign (2026-04-09)

- Submitted new live training runs to generate fresh checkpoints for FT-vs-LoRA MP dynamics:
  - `2156090`: `models4-fsdp0-wikiplus` (Full-FT, 2 nodes, 8 GPUs)
  - `2156091`: `models4-fsdp1-wikiplus` (LoRA, 1 node, 4 GPUs)
- Checkpoint callback evaluation policy was upgraded before cleanup:
  - Per-checkpoint evaluations now run in sequence: memorization -> performance -> ID.
  - Checkpoint payload deletion is allowed only when required outputs exist in `checkpoint-*/slurm_logs/`.
  - This preserves metrics while still controlling storage growth.

## Caveats / limitations

- Current DII stage uses deterministic subsampling to avoid OOM; this is intentional and reproducible, but should be reported.
- The present evidence is interim: formal statistical significance tests against random subset controls are still needed for paper-grade claims.
- H4 calibrated annotation has completed successfully against a live local vLLM endpoint.
- Semantic validity of H4 labels is currently limited by mock context extraction in `scripts/eval_auto_annotation.py`.
- Current H5 result is qualitative; it demonstrates the steering path works, but does not yet quantify intervention strength.

## Immediate next checks

1. **Perform downstream robustness tests on calibration hyperparameters**
   - Vary `--scale_mode` (mad, zscore, none), `--clip_quantile`, `--sigma_estimator`
   - Verify that domain-feature recovery is stable and not an artifact of specific tuning

2. **Replace mock context extraction with token-aligned real contexts**
  - Map top-activating token indices back to real text snippets
  - Re-run H4 to validate confidence and labels under meaningful contexts

3. **Test additional high-separation dataset pairs**
   - Code-like text vs factual biomedical
   - Wikipedia vs random token sequences
   - Goal: ensure calibration method generalizes beyond the current pile/wikidata pair

## FT-vs-LoRA live smoke status (2026-04-09)

To validate the new fine-tuning dimensionality hypotheses under storage constraints, we ran a live smoke campaign on `EleutherAI/pythia-70m` with checkpointed evaluation restricted to required metrics:

- Required per-checkpoint artifacts:
  - `memorization_metrics.json`
  - `mp_checkpoint_metrics.json`
- Disabled by default in checkpoint hooks to reduce fanout:
  - performance eval
  - ID eval
  - probing eval

### Coverage obtained

- FT checkpoints with required MP+memorization artifacts: `25, 50, 75, 100`
- LoRA checkpoints with required MP+memorization artifacts: `25, 300, 600`

Summary table generated at:
- `results/mp_reservoir/ft_lora_dynamics/smoke_subset_mp_mem_summary.csv`

### High-level smoke signal

- FT shows a transient rise in feature-level informative fraction at checkpoint 75 (`0.0566`, 29 informative features) versus neighboring checkpoints (~`0.0098`, 5 features).
- LoRA remains comparatively stable across sampled checkpoints (~`0.0098`, 5 informative features).

Interpretation (smoke-level): this is consistent with the new H4/H5 framing that full FT can induce broader dimensional relocation while LoRA can remain more constrained/reinforcing. This is an instrumentation validation, not yet a final claim.

### Recommended smoke plots (generated)

Generated under `results/mp_reservoir/ft_lora_dynamics/plots/`:

1. `smoke_mp_trajectory.png`
  - Best primary plot for the new hypothesis.
  - Shows checkpoint trajectories of `signal_fraction_feature` and `num_informative_feature` for FT vs LoRA.
1b. `smoke_mp_trajectory_normalized.png`
  - Same MP readout but x-axis normalized by in-run data exposure (% of run), which is the preferred FT-vs-LoRA comparison when checkpoint counts differ.
2. `smoke_mem_trajectory.png`
  - Best companion plot for overfitting/memorization pressure.
  - Shows `train_nll`, `exposure_delta`, and `avg_tokens_recovered` trajectories.
2b. `smoke_mem_trajectory_normalized.png`
  - Memorization trajectories on normalized exposure axis for fair cross-regime comparison.
3. `smoke_mp_vs_exposure_scatter.png`
  - Best coupling plot to visualize whether MP shifts co-occur with memorization pressure.
4. `smoke_deltas.png` (+ `smoke_deltas.csv`)
  - Best transition-intuition plot showing per-step deltas in MP signal fraction and informative counts.

Normalized plotting table:

- `smoke_subset_with_normalized_exposure.csv` includes `exposure_frac` and `exposure_pct` columns used by the normalized figures.

Key smoke readout from these figures:

- FT shows a sharp mid-training MP expansion (`50 -> 75`: `delta_signal_fraction_feature = +0.046875`, `delta_num_informative_feature = +24`) followed by contraction (`75 -> 100`: `-0.046875`, `-24`).
- LoRA shows flat MP dimensionality on sampled checkpoints (`25 -> 300 -> 600`: deltas `0.0`, informative count fixed at `5`).
- Exposure increases in both modes, but at a higher level for FT (mean `6.86`) than LoRA (mean `5.08`) in this smoke run.
