# Models4 Training Pipeline - Memorization-Only Evaluation

## Overview
Training pipeline for Llama-3.1-8B on wikiplus dataset with memorization-only evaluation at each checkpoint. This setup is designed to track memorization evolution during training without the overhead of performance and ID evaluations.

## Key Parameters
- **Dataset**: wikiplus only (data3/wikidata_Mis7.csv)
- **Epochs**: 5 (~2000 total steps for full dataset)
- **Checkpoints**: Every 130 steps (~3 per epoch, ~15 total)
- **Learning Rate**: 1e-4 (LoRA), 1e-5 (Full-FT)
- **Training Types**: Full Fine-Tuning (fsdp-0 with DeepSpeed) and LoRA (fsdp-1 with FSDP)
- **Batch Configuration**:
  - Full-FT: batch_size=1, grad_accum=2, 8 GPUs (2 nodes × 4) → effective_batch=16
  - LoRA: batch_size=4, grad_accum=2, 4 GPUs (1 node × 4) → effective_batch=32
- **Evaluation**: Memorization metrics only (discoverable, distributional, intrinsic dimension)
- **Cleanup**: Checkpoint model files deleted after evaluation (keeps slurm_logs/ with results)

## Directory Structure
```
models4/
├── Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4/
│   ├── training.out
│   ├── training.err
│   ├── checkpoint-68/
│   │   └── slurm_logs/
│   │       ├── memorization_evaluation.out
│   │       ├── memorization_evaluation.err
│   │       └── memorization_results.json
│   ├── checkpoint-136/
│   ├── checkpoint-205/
│   └── ...
└── Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/
    └── (same structure)
```

## Scripts

### 1. Launch Training
**File**: `scripts/launch_models4_wikiplus_training.sh`
- Submits 2 training jobs (fsdp-0 and fsdp-1)
- Excludes problematic nodes (node044, node041)
- Automatically triggers memorization evaluation at each checkpoint

**Usage**:
```bash
bash /home/mmahaut/projects/paramem/scripts/launch_models4_wikiplus_training.sh
```

### 2. Checkpoint Evaluation
**File**: `scripts/launch_memorization_checkpoint_eval.sh`
- Called automatically by training pipeline after each checkpoint save
- Runs only memorization evaluation (skips performance/ID)
- Deletes checkpoint model files after evaluation to save space
- Preserves slurm_logs/ directory with results

**Environment Variables**:
- `CHECKPOINT_PATH`: Path to checkpoint directory
- `USE_LORA`: 0 for full-ft, 1 for LoRA
- `KEEP_CHECKPOINT`: Set to 1 to preserve checkpoint files (optional)

### 3. Monitor Progress
**File**: `scripts/monitor_models4.sh`
- Shows active training and evaluation jobs
- Lists all checkpoints and their evaluation status
- Displays completion status with visual indicators

**Usage**:
```bash
bash /home/mmahaut/projects/paramem/scripts/monitor_models4.sh

# Or watch continuously:
watch -n 30 bash /home/mmahaut/projects/paramem/scripts/monitor_models4.sh
```

## Memorization Metrics Computed

Each checkpoint evaluation computes three types of memorization metrics:

### 1. Discoverable Memorization
- **Extractability Rate**: % of sequences extractable via prompting
- **Token Recovery Rate**: % of tokens correctly predicted
- Based on: Carlini et al. (2022) - "Quantifying Memorization Across Neural Language Models"

### 2. Distributional Memorization
- **N-gram Correlation**: Pearson correlation between training n-gram frequencies and model predictions
- Measures if model's probability distribution matches training data distribution
- Based on: Biderman et al. (2024) - "Emergent and Predictable Memorization in Large Language Models"

### 3. Intrinsic Dimension (ID)
- **Embedding ID**: Dimensionality of embedding manifold using TwoNN estimator
- Hypothesis: Higher ID → harder to memorize (more complex representation)
- Based on: L2M2 (2025) - "Understanding Memorization Through Intrinsic Dimensionality"

## Expected Timeline

With checkpoint saves every 130 steps over ~2000 total steps:
- **Checkpoints**: ~15 total (130, 260, 390, 520, 650, 780, 910, 1040, 1170, 1300, 1430, 1560, 1690, 1820, 1950)
- **Distribution**: ~3 checkpoints per epoch across 5 epochs

**Total**: ~15 checkpoints per model × 2 models = **30 evaluation points**

## Analysis After Training

Once training completes, use the existing analysis script:

```bash
cd /home/mmahaut/projects/paramem
srun --ntasks=1 poetry run python scripts/analysis/analyze_memorization_all_checkpoints.py --model-dir models4
```

This will generate:
- `memorization_evolution.png`: Metrics vs training step
- `memorization_id_correlation.png`: ID vs memorization relationship
- `memorization_lora_comparison.png`: LoRA vs full-ft comparison
- `memorization_summary.png`: Table of results
- `memorization_results.csv`: Raw data

## Comparison to Models3

**Models3**:
- Multiple datasets (wikiplus, pile, mmlu)
- All evaluations (performance, ID, probing, memorization)
- Checkpoint cleanup after all evaluations
- Result: Only 2/69 checkpoints preserved

**Models4**:
- Single dataset (wikiplus)
- Memorization-only evaluation
- Faster evaluation cycle
- More frequent checkpoints (3 per epoch vs variable in models3)
- Result: Clean evolution tracking with sufficient data points

## Troubleshooting

### Check Training Logs
```bash
tail -f /home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4/training.out
```

### Check Evaluation Logs
```bash
tail -f /home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4/checkpoint-68/slurm_logs/memorization_evaluation.out
```

### Verify Results
```bash
ls -lh /home/mmahaut/projects/paramem/models4/*/checkpoint-*/slurm_logs/memorization_results.json
```

### Re-run Evaluation on Specific Checkpoint
```bash
export CHECKPOINT_PATH=/home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4/checkpoint-68
export USE_LORA=0
export KEEP_CHECKPOINT=1  # Preserve checkpoint after eval
bash /home/mmahaut/projects/paramem/scripts/launch_memorization_checkpoint_eval.sh
```

## Notes

1. **Node Exclusions**: node044 and node041 are excluded due to known issues
2. **Resource Requirements**: 
   - Training: 4 GPUs, 400G memory
   - Evaluation: 1 GPU, 100G memory, 2-hour time limit
3. **Cleanup Behavior**: Checkpoint model files deleted after evaluation unless `KEEP_CHECKPOINT=1` is set
4. **SLURM Integration**: Uses `srun` for compute tasks, never runs Python on login node
