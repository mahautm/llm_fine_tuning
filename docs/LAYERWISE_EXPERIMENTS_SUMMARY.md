# Summary: New Layerwise Analysis Experiments

## Overview

I've created two comprehensive experiments in `/home/mmahaut/projects/paramem/paramem/` for analyzing layer representations across different fine-tuning methods (Full FT, LoRA FT, and Original models).

## Experiment 1: Layerwise Representation Comparison

**File**: `paramem/layerwise_representation_comparison.py`

**Purpose**: Compare how layer representations differ between models using:

### Metrics
1. **Information Imbalance (II)**: 
   - Extends existing `get_II.py` functionality
   - Measures information loss when moving between representations
   - Computed bidirectionally: II(A→B) and II(B→A)
   - Uses DADApy's `return_information_imbalace()` method

2. **Neighbourhood Overlap (NO)** [NEW]:
   - Computes Jaccard similarity of k-nearest neighbors
   - Range: [0, 1] where 1 = identical neighborhoods
   - Symmetric metric showing structural similarity
   - Measures preservation of local geometry

### Key Features
- All-to-all layer comparisons (creates heatmaps)
- Pairwise model comparisons (Original vs Full FT, Original vs LoRA, Full FT vs LoRA)
- Diagonal extraction plots (same layer across models)
- Activation clipping to remove outliers (0.05-0.95 quantiles)
- Configurable number of neighbors for overlap computation

### Outputs
- CSV files with II and NO metrics for each layer pair
- Heatmaps: `heatmap_II_AB_*.png`, `heatmap_II_BA_*.png`, `heatmap_NO_*.png`
- Diagonal comparisons: `diagonal_II_*.png`, `diagonal_NO_*.png`

## Experiment 2: Layerwise Probing

**File**: `paramem/layerwise_probing.py`

**Purpose**: Evaluate how well individual layer representations encode task-relevant information

### Method
- **Linear Probes**: Logistic regression on frozen layer representations
- **Single Layer Training**: Each layer probed independently
- **Cross-validation**: 5-fold CV for robust estimates
- **Multiple Benchmarks**: MMLU, ARC, HellaSwag, GSM8K, TruthfulQA, Winogrande, OpenBookQA, Lambada

### Key Features
- Layer-by-layer probing (identifies most informative layers)
- Comparison across model types (Original, Full FT, LoRA FT)
- Standardized features (StandardScaler)
- Train/test split (80/20) with stratification
- Metrics: accuracy, CV mean, CV std

### Outputs
- CSV files: `probing_results_*.csv` with per-layer accuracy
- Line plots: `probing_curves_*.png` showing accuracy across layers
- Heatmaps: `probing_heatmap_*.png` comparing models and layers
- Bar charts: `probing_best_*.png` showing best layer performance

## Running the Experiments

### Quick Start

```bash
# Experiment 1: Representation Comparison
sbatch scripts/run_layerwise_comparison.sh

# Experiment 2: Probing
sbatch scripts/run_layerwise_probing.sh
```

### Manual Execution

```bash
# Experiment 1
python paramem/layerwise_representation_comparison.py \
    --dataset pile \
    --limit 2500 \
    --k-neighbors 50

# Experiment 2
python paramem/layerwise_probing.py \
    --dataset pile \
    --benchmarks mmlu arc hellaswag gsm8k \
    --max-samples 5000
```

## File Structure

```
paramem/
├── layerwise_representation_comparison.py  # Experiment 1
├── layerwise_probing.py                    # Experiment 2
├── LAYERWISE_EXPERIMENTS_README.md         # Detailed documentation
├── get_II.py                              # Original II code (basis for Exp 1)
└── probing_task_extract_final_representations.py  # Original extraction

scripts/
├── run_layerwise_comparison.sh            # SLURM script for Exp 1
└── run_layerwise_probing.sh              # SLURM script for Exp 2

output/
├── layerwise_comparison/
│   ├── pile/
│   │   ├── heatmap_*.png
│   │   ├── diagonal_*.png
│   │   └── *.csv
│   └── wikiplus/
│       └── ...
└── layerwise_probing/
    ├── pile/
    │   ├── mmlu/
    │   │   └── probing_results_*.csv
    │   ├── probing_curves_*.png
    │   ├── probing_heatmap_*.png
    │   └── probing_best_*.png
    └── wikiplus/
        └── ...
```

## Expected Input Data

Both experiments require:

1. **Layer Representations** (pickle files in `hlayer/`):
   - Format: `{layer_id: [sample_representations]}`
   - Naming: `*{dataset}*{model_type}*.pickle`
   - Examples:
     - `Llama-pile-original.pickle`
     - `Llama-pile-full-ft.pickle`
     - `Llama-pile-lora-ft.pickle`

2. **Benchmark Data** (for Experiment 2, in `benchmark/`):
   - Format: TSV with `id\tlabel\ttext`
   - Examples: `mmlu.tsv`, `arc_dev.tsv`, etc.

## Key Differences from Existing Code

### From `get_II.py`:
- ✅ Kept: Core II computation using DADApy
- ✅ Kept: Activation clipping
- ➕ Added: Neighbourhood Overlap metric
- ➕ Added: Systematic pairwise model comparisons
- ➕ Added: Diagonal extraction and visualization
- ➕ Added: More flexible file matching

### From `probing_task_extract_final_representations.py`:
- ✅ Kept: Representation extraction logic (can be used separately)
- ➕ Added: Complete probing pipeline with training
- ➕ Added: Cross-validation and robust evaluation
- ➕ Added: Multi-benchmark support
- ➕ Added: Visualization suite

## Resource Requirements

### Experiment 1:
- Memory: ~100GB (for 2500 samples)
- Time: ~2-4 hours per dataset
- GPU: 1 (for distance computation acceleration)

### Experiment 2:
- Memory: ~100GB
- Time: ~4-8 hours (all benchmarks)
- GPU: 1 (for model loading if extracting fresh representations)

## Scientific Questions Addressed

### Experiment 1 (Representation Comparison):
1. **How much do fine-tuning methods change representations?**
   - Compare II values between Original vs Full FT vs LoRA
   
2. **Are changes layer-specific or global?**
   - Analyze diagonal values and heatmap patterns
   
3. **Do LoRA and Full FT make similar changes?**
   - Compare Full FT vs LoRA directly
   
4. **Is local geometry preserved?**
   - Use Neighbourhood Overlap to check structural preservation

### Experiment 2 (Probing):
1. **Which layers are most informative for each task?**
   - Identify peak layers in probing curves
   
2. **Does fine-tuning improve task representations?**
   - Compare Original vs FT probing accuracy
   
3. **Where do LoRA and Full FT differ in learned representations?**
   - Compare layer-wise probe patterns
   
4. **Do different benchmarks use different layers?**
   - Compare peak layers across MMLU, ARC, etc.

## Next Steps

1. **Generate representations** if not already available:
   ```bash
   python paramem/probing_task_extract_final_representations.py \
       meta-llama/Llama-3.2-1B-Instruct \
       16 \
       benchmark/mmlu.tsv \
       hlayer/Llama-pile-original
   ```

2. **Run experiments** using SLURM scripts

3. **Analyze results**:
   - Check heatmaps for layer-specific changes
   - Compare diagonal plots across model pairs
   - Identify peak probing layers for each benchmark
   - Look for patterns (e.g., early vs late layer changes)

4. **Extend** (optional):
   - Add more metrics (CKA, PWCCA)
   - Try nonlinear probes (MLP)
   - Analyze across training checkpoints
   - Add statistical significance tests

## Documentation

Full documentation available in:
- `/home/mmahaut/projects/paramem/paramem/LAYERWISE_EXPERIMENTS_README.md`

## Dependencies

Already available in `paramem` environment:
- ✅ `dadapy` (for II)
- ✅ `scikit-learn` (for probing)
- ✅ `torch`, `transformers`
- ✅ `matplotlib`, `seaborn`
- ✅ `numpy`, `pandas`
- ✅ `typer`, `tqdm`
