# Implementation Summary: Memorization Metrics

## What Was Done

I've implemented **minimal changes** to your codebase to enable testing of memorization metrics as suggested by your colleague. The implementation is based on three key papers:

### 1. Papers Referenced

- **Discoverable Memorization**: [Carlini et al. 2022](https://arxiv.org/abs/2202.07646)
  - Tests if training sequences can be extracted via prompting
  
- **Distributional Memorization**: [Biderman et al. 2024](https://arxiv.org/pdf/2407.14985)
  - Measures correlation with n-gram model outputs
  
- **Intrinsic Dimension & Memorization**: [L2M2 Workshop 2025](https://aclanthology.org/2025.l2m2-1.2/)
  - Shows high ID sequences are harder to memorize

## Files Created

1. **`paramem/memorization_metrics.py`** (Core implementation)
   - `compute_discoverable_memorization()` - Extract sequences via prompting
   - `compute_distributional_memorization()` - N-gram correlation
   - `compute_embedding_intrinsic_dimension()` - ID via TwoNN method
   - `compute_all_memorization_metrics()` - Main entry point

2. **`scripts/analysis/analyze_memorization_evolution.py`** (Analysis script)
   - Processes all checkpoints in a training run
   - Creates plots: memorization vs training step
   - Tests hypothesis: high ID → harder to memorize

3. **`scripts/tests/test_memorization_metrics.py`** (Quick test)
   - Tests metrics on single checkpoint
   - Minimal data requirements (50 sequences)
   - Fast validation of implementation

4. **`MEMORIZATION_METRICS_README.md`** (Documentation)
   - Usage instructions
   - Expected results
   - Integration guide

## How to Use

### Quick Test (Recommended First Step)

```bash
# Test on one checkpoint with minimal data
srun --ntasks=1 python scripts/tests/test_memorization_metrics.py \\
    --checkpoint models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4/checkpoint-100 \\
    --data data3/wikidata_Mis7.csv
```

### Full Analysis

```bash
# Analyze entire training run
srun --ntasks=1 python scripts/analysis/analyze_memorization_evolution.py \\
    models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4 \\
    --data-file data3/wikidata_Mis7.csv \\
    --output-dir memorization_results \\
    --num-samples 200
```

## Integration with Existing Code

The implementation:
- ✅ Uses the same data format as your existing evaluation (`data3/*.csv`)
- ✅ Works with your checkpoint structure (`checkpoint-*/`)
- ✅ Can be called alongside your existing metrics (`exact_match`, etc.)
- ✅ Requires no changes to training code
- ✅ Minimal dependencies (uses your existing transformers setup)

### Example Integration

```python
# In your existing evaluation loop:
from paramem.memorization_metrics import compute_all_memorization_metrics

# After computing exact_match, paraphrase stability, etc.
mem_metrics = compute_all_memorization_metrics(
    checkpoint_path=checkpoint,
    training_sequences=training_data['query'].tolist()
)

# Now you have:
# - mem_metrics['extractability_rate']  
# - mem_metrics['intrinsic_dimension']
# - mem_metrics['ngram_correlation']
```

## What to Expect

### Plots Generated

1. **Memorization Evolution**: 4-panel plot showing all metrics vs training step
2. **ID vs Memorization**: Scatter plot testing the main hypothesis
3. **Combined Score**: Normalized metrics on single plot

### Key Questions to Answer

1. **Does memorization increase during training?** (Should see upward trend)
2. **Do high-ID sequences memorize less?** (Should see negative correlation)
3. **Which metric is most informative?** (Discoverable vs distributional vs ID)
4. **How does this compare to exact_match?** (Your existing metric)

## Minimal Dependencies

The implementation only requires:
- `torch` (already have)
- `transformers` (already have)
- `scikit-learn` (for ID computation - add with: `pip install scikit-learn`)
- `matplotlib`, `seaborn` (for plots - already have)

## Next Steps

1. **Run quick test** to verify implementation works
2. **Analyze one training run** end-to-end
3. **Compare** with your existing metrics
4. **Iterate** on metrics if needed (adjust parameters, try variations)

## Notes

- The implementation samples sequences for efficiency (default: 200)
- ID computation uses TwoNN estimator (standard method)
- N-gram model uses trigrams by default (can adjust)
- All metrics are normalized to [0, 1] for comparison

---

**Ready to test!** Start with `scripts/tests/test_memorization_metrics.py` on any checkpoint you have available.
