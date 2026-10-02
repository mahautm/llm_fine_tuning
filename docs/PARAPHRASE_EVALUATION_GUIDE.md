# Paraphrase-Based Stability Evaluation - Integration Guide

## Overview

The paraphrase-based stability evaluation has been integrated into the checkpoint evaluation pipeline. This evaluation tests how consistently the model answers the same factual question when phrased differently, inspired by Amazon's factual-confidence-of-llms research.

## How It Works

### Pipeline Flow

```
New Checkpoint Created
    ↓
launch_checkpoint_tests.sh
    ↓
performance_evaluation.py
    ↓
evaluate_wikidata_with_paraphrases()
    ↓
Results saved to checkpoint/slurm_logs/paraphrase_stability.json
```

### Paraphrase Caching

**Key Feature**: Paraphrases are generated once and cached to avoid regeneration for each checkpoint.

- **Cache File**: `/home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json`
- **Shared Across**: All checkpoints, all training runs
- **Behavior**:
  - First checkpoint: Generates paraphrases and saves to cache
  - Subsequent checkpoints: Loads from cache (much faster)
  - New questions: Generates and appends to cache

## Integration Details

### 1. performance_evaluation.py

Added automatic paraphrase evaluation after standard knowledge evaluation:

```python
# Evaluate with paraphrase-based stability
from paramem.evaluation.wikidata_paraphrase_eval import evaluate_wikidata_with_paraphrases

# Set up paths
paraphrase_output = os.path.join(checkpoint_path, "slurm_logs", "paraphrase_stability.json")
paraphrase_cache = "/home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json"

results.update(evaluate_wikidata_with_paraphrases(
    wikidata_path, model, tokenizer, device,
    n_samples=200,  # Evaluate 200 questions
    num_paraphrases=5,  # 5 paraphrases per question
    use_llm_paraphrasing=False,  # Use rule-based (faster)
    output_file=paraphrase_output,
    paraphrase_cache_file=paraphrase_cache
))
```

### 2. wikidata_paraphrase_eval.py

Enhanced with caching mechanism:

- **New parameter**: `paraphrase_cache_file` (optional)
- **Cache loading**: Loads existing cache at start
- **Cache updates**: Saves new paraphrases after evaluation
- **Fallback**: Works without cache (generates on-the-fly)

### 3. launch_checkpoint_tests.sh

No changes needed! The script already calls `performance_evaluation.py`, which now includes paraphrase evaluation.

## Metrics Generated

The evaluation adds these metrics to the checkpoint results:

```json
{
  "wikidata_paraphrase_overall_accuracy": 0.754,
  "wikidata_paraphrase_consistency_rate": 0.823,
  "wikidata_paraphrase_acc_when_consistent": 0.892,
  "wikidata_paraphrase_acc_when_inconsistent": 0.341,
  "wikidata_paraphrase_num_consistent": 165,
  "wikidata_paraphrase_num_questions": 200
}
```

### Metric Definitions

- **overall_accuracy**: Average accuracy across all questions and paraphrases
- **consistency_rate**: Fraction of questions where all paraphrases get the same answer
- **acc_when_consistent**: Accuracy on questions where model is consistent
- **acc_when_inconsistent**: Accuracy on questions where model varies its answer

### Interpretation

- High consistency + high accuracy = Strong, stable knowledge
- High consistency + low accuracy = Consistently wrong (systematic error)
- Low consistency + high accuracy = Unstable despite correct answers
- Low consistency + low accuracy = Weak, unreliable knowledge

## Output Files

### Per-Checkpoint Results

Location: `{checkpoint_path}/slurm_logs/paraphrase_stability.json`

Structure:
```json
{
  "metrics": {
    "wikidata_paraphrase_overall_accuracy": 0.754,
    ...
  },
  "per_question_results": [
    {
      "question": "What is the capital of France?",
      "expected_answers": ["Paris"],
      "paraphrases": ["What is France's capital?", ...],
      "predictions": ["Paris", "Paris", "Paris", ...],
      "accuracy": 1.0,
      "is_consistent": true
    },
    ...
  ]
}
```

### Shared Paraphrase Cache

Location: `/home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json`

Structure:
```json
{
  "What is the capital of France?": [
    "What is France's capital?",
    "The capital of France is what?",
    "France's capital city is what?",
    "What city serves as France's capital?",
    "Which city is the capital of France?"
  ],
  ...
}
```

## Paraphrasing Methods

### Rule-Based (Default)

Fast, deterministic paraphrasing using pattern matching:

**Patterns**:
- "What can be categorized as X?" → "What is categorized as X?", "X is what?", etc.
- "The X of Y is Z" → "What is the X of Y?", "Y's X is what?", etc.
- "X was created by Y" → "Who created X?", "X was made by whom?", etc.
- "X is located in Y" → "Where is X located?", "X is in what location?", etc.

### LLM-Based (Optional)

Use model to generate natural paraphrases (set `use_llm_paraphrasing=True`):

**Pros**: More natural, diverse paraphrases
**Cons**: Slower, requires model inference, non-deterministic

## Performance Considerations

### Evaluation Time

- **First checkpoint**: ~5-10 minutes (generates + caches paraphrases)
- **Subsequent checkpoints**: ~2-3 minutes (uses cached paraphrases)
- **Per question**: ~0.6 seconds (6 model inferences: 1 original + 5 paraphrases)

### Sample Size

Current configuration: 200 questions (1200 total inferences)

To adjust:
```python
n_samples=200  # Change this value in performance_evaluation.py
```

### Number of Paraphrases

Current configuration: 5 paraphrases per question

To adjust:
```python
num_paraphrases=5  # Change this value in performance_evaluation.py
```

## Monitoring Progress

During evaluation, you'll see:

```
============================================================
Paraphrase-Based Stability Evaluation
============================================================
Questions: 200
Paraphrases per question: 5
Paraphrasing method: Rule-based
Cache available: 0 questions
============================================================

Evaluating with paraphrases: 100%|██████████| 200/200 [02:15<00:00,  1.48it/s]

✅ Saved 200 new paraphrases to cache (total: 200)

============================================================
Paraphrase Stability Results
============================================================
Overall Accuracy: 0.754
Consistency Rate: 0.823 (165/200)
Accuracy when Consistent: 0.892
Accuracy when Inconsistent: 0.341
============================================================
```

## Troubleshooting

### Cache Not Loading

**Symptom**: Every checkpoint generates paraphrases from scratch

**Solution**: Check cache file permissions and path:
```bash
ls -la /home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json
```

### Slow Evaluation

**Symptom**: Paraphrase evaluation takes >10 minutes per checkpoint

**Causes**:
1. Cache not being used (check logs for "Loading cached paraphrases")
2. Too many samples (reduce `n_samples`)
3. LLM-based paraphrasing enabled (switch to rule-based)

### Missing Metrics

**Symptom**: Paraphrase metrics not in results

**Solution**: Check that evaluation completed successfully:
```bash
grep "Paraphrase Stability Results" {checkpoint}/slurm_logs/slurm-*.out
```

## Future Enhancements

Potential improvements:

1. **LLM Paraphrasing**: Switch to model-generated paraphrases for better quality
2. **Cross-Dataset**: Extend to other datasets beyond wikidata
3. **Temporal Analysis**: Track consistency changes across training
4. **Confidence Scores**: Add model confidence to predictions
5. **Semantic Similarity**: Measure paraphrase quality

## Testing

To verify integration is working:

```bash
# Check integration
grep -A 5 "evaluate_wikidata_with_paraphrases" \
  /home/mmahaut/projects/paramem/paramem/evaluation/performance_evaluation.py

# Check for existing cache
ls -lh /home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json

# Run a checkpoint evaluation and monitor
tail -f models2/{model}/checkpoint-{N}/slurm_logs/slurm-*.out
```

## References

- Amazon's factual-confidence-of-llms: https://github.com/amazon-science/factual-confidence-of-llms
- Paper: "Generating Benchmarks for Factual Accuracy Evaluation of Language Models"

---

Last Updated: December 19, 2025
