# MP-Conditional Dataset Partitioning: Task List & Experimental Pipeline
> **Working title**: *"The Uncommitted Reservoir: Dataset-Conditional Marchenko-Pastur Partitioning of LLM Representation Space"*
> **Target**: NeurIPS / ICLR / ICML (top-tier ML conference)
> Last updated: 2026-03-23

---

## 0. Core Hypotheses (anchor everything here)

| # | Hypothesis | Falsifiable prediction |
|---|---|---|
| H1 | Singular directions of a weight matrix W are conditionally signal/noise depending on data distribution D | MP bulk edge λ_max shifts measurably across datasets at fixed layer |
| H2 | Directions polysemantically encoding multiple concepts appear as one above-bulk spike in the weight spectrum but decompose under dataset conditioning | A direction above-bulk for D_all splits into one signal / one noise direction when D is restricted to a subdomain |
| H3 | Fine-tuning is targeted crystallization: new-domain directions were near-threshold (not deep-bulk) in the base model | Base model shows a band of near-threshold directions that overlap with fine-tuning top singular directions |
| H4 | The MP bulk is a functional reservoir: ablating in-bulk directions harms cross-domain generalization more than above-bulk ablations | Performance delta from bulk ablation > above-bulk ablation on held-out domains |
| H5 | Generalization distance between two datasets is predictable from their signal-subspace overlap | DII(S(D_i), S(D_j)) correlates with cross-dataset transfer performance |

---

## 1. Literature & Framing (Week 1–2)

### 1.1 Must-read papers
- [ ] Martin & Mahoney — Implicit Self-Regularization (JMLR 2021) — especially §4 phases
- [ ] arXiv:2410.17770 — Locating Information via RMT (ICLR 2025)
- [ ] Anthropic SAE paper (Templeton et al. 2024) — superposition and feature geometry
- [ ] ASVD (Yuan et al. 2023, arXiv:2312.05821) — activation-aware SVD baseline
- [ ] Laio — Information Imbalance (2021) + Differential Information Imbalance (2023)
- [ ] Mézard & Montanari — *Information, Physics, Computation* ch.5 (replica / cavity for MP intuition)
- [ ] Simsekli et al. (2020) — Hausdorff dimension, tail index, generalization bounds
- [ ] Heavy-Tailed Self-Regularization (Martin & Mahoney ICML 2019) — 5+1 phases

### 1.2 Related work table to write
- [ ] Map: SAE approaches vs. spectral approaches vs. probing approaches (conceptual 3-way comparison)
- [ ] Identify the exact gap: no prior work does (MP-null × dataset-conditioned × polysemanticity decomposition) jointly
- [ ] Draft 1-paragraph "what we do that nobody has done" for paper intro

---

## 2. Codebase Setup (Week 1–2)

### 2.1 Repository structure
```
mp-reservoir/
├── configs/           # Hydra configs for all experiments
├── data/              # Dataset loading, tokenization, activation caching
├── spectral/          # Core MP fitting + threshold logic
├── sae/               # SAE baseline + MP-gated SAE variant
├── probing/           # DII + information imbalance utilities
├── ablation/          # Knock-out and direction ablation evals
├── scripts/           # Slurm launchers
├── notebooks/         # Exploratory / figure generation
└── evals/             # Downstream task evaluation harness
```

### 2.2 Core dependencies
- [ ] `transformers` + `accelerate` — model loading (Pythia, Llama)
- [ ] `einops` — tensor manipulation for activation batching
- [ ] `sklearn` (PCA, covariance) or `torch.linalg.eigh` — covariance eigendecomposition
- [ ] `scipy.stats` — MP distribution fitting (Marchenko-Pastur is not in standard libraries, need custom)
- [ ] Custom MP fitter — fit `σ²` and aspect ratio `γ = p/n` to empirical ESD via moment matching or MLE
- [ ] `faiss` — k-NN for DII / information imbalance
- [ ] `datasets` (HuggingFace) — dataset pipeline
- [ ] `submitit` — Slurm job submission from Python
- [ ] `wandb` or `mlflow` — experiment tracking
- [ ] `pytest` — unit tests for spectral utilities

### 2.3 Key custom modules to implement

#### `spectral/mp_fit.py`
- [ ] `fit_mp(eigenvalues, gamma) -> (sigma2, lambda_max, lambda_min)` — moment matching
- [ ] `fit_mp_mle(eigenvalues) -> (sigma2, gamma)` — MLE variant
- [ ] `partition(eigenvalues, lambda_max) -> (signal_mask, noise_mask)`
- [ ] `plot_esd(eigenvalues, fitted_mp)` — diagnostic visualization
- [ ] Unit tests: verify recovery on synthetic Wishart + spike matrices

#### `spectral/activation_cov.py`
- [ ] `collect_activations(model, dataloader, layer_ids, n_samples) -> dict[layer -> Tensor]`
- [ ] `compute_cov(activations, center=True) -> Tensor` — handle `n < d` (tall vs. wide regimes)
- [ ] `project_onto_singular_basis(cov, W_svd) -> projected_eigenvalues`
- [ ] Note: use online/incremental covariance for large n to avoid OOM

#### `spectral/dataset_partition.py`
- [ ] `DatasetPartition(model, datasets, layers)` — main orchestrator class
- [ ] `compute_all_partitions() -> Tensor[n_datasets, n_layers, n_directions]` — the core 3D tensor
- [ ] `compare_partitions(D_i, D_j) -> subspace_overlap_score`
- [ ] Cache results to disk (hdf5 or safetensors)

#### `sae/mp_gated_sae.py`
- [ ] Standard SAE baseline — encoder/decoder with L1 penalty
- [ ] MP-gated SAE variant — replace fixed ReLU threshold with dataset-conditional MP threshold
- [ ] `FeatureActivityMatrix` — `features × datasets` binary map output
- [ ] Polysemanticity detector — flag directions that split on dataset conditioning (above bulk for D_all, mixed for D_subsets)

#### `probing/dii.py`
- [ ] `information_imbalance(X, Y, k=10) -> float` — standard II
- [ ] `differential_ii(X, Y, direction_weights) -> float` — differentiable, Laio 2023 variant
- [ ] `subspace_ii(activations, signal_mask, full=True) -> float` — compare signal-subspace distances to full-space distances
- [ ] `optimize_direction_weights(activations, target) -> weights` — learn soft MP threshold via DII gradient

---

## 3. Data Pipeline (Week 2–3)

### 3.1 Dataset selection
Choose datasets to maximize distributional diversity while keeping compute tractable:

| Dataset | Domain | Rationale |
|---|---|---|
| The Pile / C4 | General English | Pretraining distribution — baseline |
| GitHub Code (Python) | Code | Structural + lexical contrast |
| PubMed abstracts | Biomedical | Domain-specific terminology |
| CC-100 French | Non-English | Cross-lingual test |
| Math (MATH dataset) | Formal symbolic | Extreme structural contrast |
| Spoken transcripts (e.g. OpenSubtitles) | Informal/spoken | Register contrast |
| Random tokens | — | Pure noise baseline (should be full MP always) |

### 3.2 Data preprocessing
- [ ] Tokenize all datasets with same tokenizer (model-matched)
- [ ] Sample `n = 5000` sequences of fixed length `L = 128` tokens per dataset
- [ ] Cache tokenized datasets to disk
- [ ] Verify no token distribution overlap artifacts (domain leakage check)

### 3.3 Activation caching (cluster)
- [ ] Script: `scripts/cache_activations.py --model pythia-2.8b --datasets all --layers 0,4,8,12,16,20,24`
- [ ] Slurm job: 1 GPU (A100), ~2h per model × dataset combination
- [ ] Store as `activations/{model}/{layer}/{dataset}.safetensors`
- [ ] Estimated storage: ~40GB for Pythia-2.8B across 7 datasets × 7 layers

---

## 4. Experiment 1: MP Partition Stability (Week 3–4)

**Goal**: establish that the partition varies across datasets in a non-random, structured way (H1).

### Tasks
- [ ] Compute `λ_max(D, L)` for all (dataset, layer) pairs
- [ ] Compute fraction of directions in signal partition per (dataset, layer)
- [ ] Statistical test: is the partition variance across datasets > variance across random subsets of same dataset?
- [ ] Visualization: heatmap of signal fraction — rows = layers, columns = datasets

### Expected result
High-level (later) layers should show maximum dataset differentiation. Random-token baseline should be all-bulk always. General English should have many above-bulk directions; narrow-domain datasets should have fewer but more concentrated ones.

### Sanity checks
- [ ] Pythia-70m (untrained) should give near-pure MP for all datasets
- [ ] λ_max should grow monotonically from random-token baseline → general English → specific domains

---

## 5. Experiment 2: Polysemanticity Decomposition (Week 4–6)

**Goal**: identify directions that are above-bulk for D_all but split into signal/noise when datasets are separated (H2).

### Tasks
- [ ] For each layer L, compute partition on D_all (all datasets pooled)
- [ ] For each above-bulk direction k in D_all, evaluate λ_k^(D_i) for each individual D_i
- [ ] Define **polysemantic direction**: above-bulk for ≥2 datasets but below bulk for ≥1 dataset
- [ ] Define **monosemantic direction**: above-bulk for exactly 1 dataset
- [ ] Plot: per-layer counts of polysemantic vs. monosemantic vs. universal (above all) vs. dead (below all)

### MP-Gated SAE experiment
- [ ] Train standard SAE on D_all activations at layer L=12 (mid-network)
- [ ] Train MP-gated SAE on same activations — compare feature splitting behavior
- [ ] Measure: does MP-gated SAE recover more monosemantic features per polysemantic direction?
- [ ] Interpretability check: for a randomly selected polysemantic direction, manually inspect top-10 activating tokens per subdataset

---

## 6. Experiment 3: DII-Based Subspace Sufficiency (Week 5–7)

**Goal**: test whether the signal subspace S(D_i, L) preserves the neighborhood structure of D_i better than random subsets of the same dimensionality (H5).

### Tasks
- [ ] For each (D_i, L), compute II(d_signal, d_full) and II(d_random_same_size, d_full)
- [ ] Statistical test: is signal subspace significantly better than size-matched random subspace?
- [ ] Soft optimization: use differentiable DII to learn per-direction weights for each D_i — do learned weights correlate with MP signal/noise partition?
- [ ] Cross-dataset DII matrix: compute II(S(D_i), S(D_j)) for all pairs — build and visualize the 7×7 matrix per layer

### Expected result
Signal subspace should be significantly better than random for all datasets. The DII matrix should reveal clustering (code subspace ↔ math subspace close; biomedical ↔ general English close; French ↔ general English intermediate).

---

## 7. Experiment 4: Ablation — Reservoir Function of the Bulk (Week 7–9)

**Goal**: test that bulk directions are not dead weight but active reservoir for generalization (H4).

### Tasks
- [ ] Design 3 ablation types:
  - `ablate_above_bulk(D_i)` — zero out signal directions for D_i
  - `ablate_in_bulk(D_i)` — zero out a size-matched set of bulk directions
  - `ablate_random` — zero out random directions (control)
- [ ] Evaluation: measure perplexity delta on D_i (same domain) and D_j (cross-domain) after each ablation type
- [ ] Key prediction: `ablate_in_bulk` should harm D_j (generalization) more than `ablate_above_bulk`

### Cross-domain transfer test
- [ ] Fine-tune base model on D_biomedical (small LoRA)
- [ ] Check: do the LoRA top singular directions overlap with directions that were near-threshold (not deep-bulk) in the base model?
- [ ] Quantify: cosine similarity between LoRA delta directions and base-model near-threshold directions

---

## 8. Experiment 5: Scaling & Universality (Week 9–11)

**Goal**: check whether findings are universal across model scales and architectures.

### Tasks
- [ ] Replicate Exp. 1–3 on Pythia {70m, 160m, 410m, 1.4b, 2.8b} — test scaling of partition structure
- [ ] Replicate on 1 non-Pythia model (Llama-3-8B or Mistral-7B) — test cross-architecture universality
- [ ] Check: does the number of monosemantic directions grow with model scale? (should, per superposition theory)
- [ ] Check: does the near-threshold band width grow or shrink with scale?

---

## 9. Paper Structure (Draft target: Week 12)

```
1. Introduction
   - The MP bulk as uncommitted reservoir (H4 big picture)
   - Gap: no dataset-conditional treatment of spectral signal/noise
   - Contributions: 3-way (polysemanticity × dataset-conditional MP × generalization framing)

2. Background
   - Marchenko-Pastur and spectral analysis of weight matrices
   - Superposition and SAEs
   - Information imbalance

3. Framework: Dataset-Conditional MP Partitioning
   - Formal definition: signal/noise partition conditioned on D
   - Polysemanticity as above-bulk-for-all ↔ signal/noise-per-domain
   - MP-Gated SAE variant

4. Experiments
   4.1 Partition structure across datasets (Exp 1)
   4.2 Polysemanticity decomposition (Exp 2)
   4.3 Subspace sufficiency via DII (Exp 3)
   4.4 Reservoir function ablation (Exp 4)
   4.5 Scaling (Exp 5)

5. Theoretical Discussion
   - Crystallization framing: learning = progressive MP-bulk escape
   - Fine-tuning = fast near-threshold promotion
   - Generalization = signal-subspace overlap
   - The tail never shuts up: optimizer information-theoretic constraint

6. Related Work

7. Conclusion
```

---

## 10. Compute Estimate (Slurm)

| Experiment | Model | Jobs | Est. GPU-hours |
|---|---|---|---|
| Activation caching | Pythia-2.8B | 49 (7 datasets × 7 layers) | ~20h A100 |
| Activation caching | Pythia scaling (5 models) | 245 | ~50h A100 |
| SAE training | Pythia-2.8B L12 | 2 (standard + MP-gated) | ~10h A100 |
| DII computation (faiss) | All datasets | 21 (7×6/2 pairs per layer) | ~5h CPU |
| LoRA fine-tuning (Exp 4) | Pythia-2.8B | 3 | ~6h A100 |
| Ablation evals | Pythia-2.8B | ~30 | ~15h A100 |
| **Total** | | | **~106h A100** |

Fits comfortably within a 200 A100-hour cluster budget. Scale up Pythia runs are mostly activation caching (CPU-bound after loading).

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| n << d in activation covariance (wide regime) | Use shrinkage estimator (Ledoit-Wolf) or work with empirical SVD directly |
| MP fit unstable for non-square activation cov | Always fit on the smaller eigenvalue set; use effective γ = min(n,d)/max(n,d) |
| Polysemantic directions indistinguishable from universal | Include random-token baseline as hard floor; universal = above MP for random tokens too |
| SAE training doesn't converge on MP-gated variant | Anneal MP threshold from standard SAE initialization; use warm-start |
| DII computation expensive for large n | Subsample to n=2000 for DII; use FAISS approximate k-NN |
| Negative result on H4 | If bulk ablation doesn't hurt generalization, reframe as "bulk is truly dead capacity" — still publishable as refutation |

---

## 12. Immediate Next Steps (This Week)

- [ ] **[Day 1]** Set up repo, install dependencies, implement `mp_fit.py` with unit tests
- [ ] **[Day 1]** Load Pythia-70m, verify activation collection works end-to-end
- [ ] **[Day 2]** Implement `activation_cov.py`, validate MP fit on Pythia-70m random init (should recover near-perfect MP)
- [ ] **[Day 2]** Write Slurm launcher for activation caching — test on 1 dataset × 1 layer
- [ ] **[Day 3]** Cache all activations for Pythia-2.8B (submit cluster jobs)
- [ ] **[Day 3]** Implement partition comparison and first heatmap visualization
- [ ] **[Day 4]** Sanity check Exp 1 results (random baseline = all bulk; general English = most signal)
- [ ] **[Day 5]** Start SAE baseline on L12 activations; verify standard SAE matches published baselines
