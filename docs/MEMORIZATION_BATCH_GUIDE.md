# Running Memorization Evaluation on Existing Checkpoints

## Quick Start

### 1. Submit Batch Jobs for All Checkpoints

```bash
cd /home/mmahaut/projects/paramem
./scripts/entrypoints/jobs/run_memorization_on_checkpoints.sh
```

This will:
- Find all checkpoints in `models3/` that have `slurm_logs/`
- Submit SLURM jobs to compute memorization metrics
- Skip checkpoints already evaluated
- Results saved to: `checkpoint-*/slurm_logs/memorization_metrics.out` and `.json`

### 2. Monitor Progress

```bash
# Check job queue
squeue -u $USER | grep mem_

# Watch a specific checkpoint
tail -f models3/Llama-3.1-8B-Instruct-fsdp-0-mmlu-lr1e-4/checkpoint-130/slurm_logs/memorization_metrics.out

# Count completed evaluations
find models3 -name "memorization_metrics.json" | wc -l
```

### 3. Analyze Results

After jobs complete:

```bash
srun --ntasks=1 python scripts/analysis/analyze_memorization_all_checkpoints.py
```

This creates:
- `memorization_analysis_all/memorization_all_checkpoints.csv` - Raw data
- `memorization_analysis_all/memorization_evolution_all.png` - Time series plots
- `memorization_analysis_all/id_vs_memorization_all.png` - Hypothesis testing
- `memorization_analysis_all/lora_vs_fullft_memorization.png` - Comparison plots
- `memorization_analysis_all/memorization_summary_table.png` - Statistics table

## What Gets Evaluated

### Models (6 total):
- Llama-3.1-8B-Instruct-fsdp-0-mmlu-lr1e-4 (Full-FT)
- Llama-3.1-8B-Instruct-fsdp-0-pile-lr1e-4 (Full-FT)
- Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4 (Full-FT)
- Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4 (LoRA)
- Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4 (LoRA)
- Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4 (LoRA)

### Checkpoints:
Only those with existing `slurm_logs/` directory (already evaluated with performance/ID)

### Training Data:
- Wikidata datasets (Mis7, Qwe7, Met7) from `data3/`
- Automatically matched based on model training dataset

### Metrics Computed:
1. **Extractability Rate** - Can sequences be extracted? (discoverable memorization)
2. **Token Recovery Rate** - Partial memorization strength
3. **N-gram Correlation** - Distributional memorization
4. **Intrinsic Dimension** - Embedding complexity (hypothesis: high ID → low memorization)

## Output Format

Results follow the same structure as existing evaluations:

```
checkpoint-130/
└── slurm_logs/
    ├── performance_evaluation.out      [existing]
    ├── ID_matrixent.out                [existing]
    ├── layerwise_probing.out          [existing]
    ├── memorization_metrics.out        [NEW]
    ├── memorization_metrics.err        [NEW]
    └── memorization_metrics.json       [NEW]
```

## Resource Requirements

Per checkpoint:
- **GPU**: 1 GPU
- **Memory**: 100G
- **Time**: ~30-60 minutes per checkpoint
- **Partition**: alien

## Troubleshooting

### Jobs failing?
Check error logs:
```bash
tail models3/*/checkpoint-*/slurm_logs/memorization_metrics.err
```

### Re-run failed checkpoints:
The script automatically skips completed checkpoints. Just run again:
```bash
./scripts/entrypoints/jobs/run_memorization_on_checkpoints.sh
```

### No data in analysis?
Ensure jobs completed:
```bash
grep -l "COMPLETED" models3/*/checkpoint-*/slurm_logs/memorization_metrics.out | wc -l
```

## Integration with Existing Analysis

The memorization metrics complement your existing evaluations:

- **Performance** (MMLU, ARC, etc.) - Task accuracy
- **ID / Matrix Entropy** - Representation complexity
- **Layerwise Probing** - Knowledge localization
- **Memorization** ← NEW - Training data retention

All can be plotted together to understand training dynamics!
