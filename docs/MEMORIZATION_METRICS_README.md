# Memorization Metrics Implementation

This implementation provides metrics to quantify memorization based on recent research:

## Metrics Implemented

### 1. **Discoverable Memorization** 
*From [Carlini et al. 2022](https://arxiv.org/abs/2202.07646)*

- **What it measures**: Can training sequences be extracted from the model?
- **How**: Prompt the model with a prefix and see if it completes with the exact training continuation
- **Metrics**:
  - `extractability_rate`: Fraction of sequences that can be fully extracted
  - `avg_tokens_recovered`: Average token-level recovery rate

### 2. **Distributional Memorization**
*From [Biderman et al. 2024](https://arxiv.org/pdf/2407.14985)*

- **What it measures**: Does the model's output distribution match training n-gram statistics?
- **How**: Compute correlation between model predictions and n-gram frequencies
- **Metric**: `ngram_correlation` - correlation coefficient

### 3. **Intrinsic Dimension (ID)**
*From [L2M2 Workshop 2025](https://aclanthology.org/2025.l2m2-1.2/)*

- **What it measures**: Complexity/diversity of learned representations
- **Hypothesis**: Higher ID → harder to memorize (sequences with high ID are more complex)
- **How**: TwoNN estimator on embedding representations
- **Metric**: `intrinsic_dimension` - estimated dimensionality

## Quick Start

### Test on a single checkpoint:

```bash
srun --ntasks=1 python scripts/tests/test_memorization_metrics.py \\
    --checkpoint models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4/checkpoint-100 \\
    --data data3/wikidata_Mis7.csv
```

### Analyze all checkpoints in a training run:

```bash
srun --ntasks=1 python scripts/analysis/analyze_memorization_evolution.py \\
    models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4 \\
    --data-file data3/wikidata_Mis7.csv \\
    --output-dir memorization_results \\
    --num-samples 200
```

This will:
- Compute metrics for each checkpoint
- Save individual results as JSON
- Create plots showing evolution over training
- Test the hypothesis: "Higher ID → Harder to memorize"

## Output

The analysis produces:

1. **CSV**: `memorization_evolution.csv` - All metrics across checkpoints
2. **Plots**:
   - `memorization_evolution.png` - All metrics over training
   - `id_vs_memorization.png` - ID vs extractability correlation
   - `combined_memorization.png` - Normalized composite score

## Usage in Your Pipeline

### Minimal integration with existing code:

```python
from paramem.memorization_metrics import compute_all_memorization_metrics

# After training or at evaluation time
metrics = compute_all_memorization_metrics(
    checkpoint_path="path/to/checkpoint",
    training_sequences=your_training_sequences
)

print(f"Extractability: {metrics['extractability_rate']}")
print(f"Intrinsic Dimension: {metrics['intrinsic_dimension']}")
```

### Use with your existing evaluation scripts:

The metrics can be computed alongside your existing `exact_match` evaluations in `slot_filling.py`. The memorization metrics use the same input format (sequences from CSV files).

## Expected Results

Based on the papers:

1. **Memorization should increase during training** (all three metrics)
2. **ID vs Memorization**: Negative correlation expected
   - High ID sequences → harder to memorize → lower extractability
3. **Distributional memorization** correlates with "expansion phase" during pretraining
4. **Discoverable memorization** is the strictest measure

## Files Created

- `paramem/memorization_metrics.py` - Core metric implementations
- `scripts/analysis/analyze_memorization_evolution.py` - Analysis script for training runs
- `scripts/tests/test_memorization_metrics.py` - Quick test on single checkpoint
- `MEMORIZATION_METRICS_README.md` - This file

## Next Steps

1. **Test the implementation** on one of your checkpoints
2. **Run full analysis** on a complete training run
3. **Compare** with your existing metrics (exact_match, paraphrase stability)
4. **Investigate**: Do sequences with high ID actually memorize less?
