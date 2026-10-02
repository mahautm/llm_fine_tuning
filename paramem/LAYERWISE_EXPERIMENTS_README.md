# Layerwise Analysis Experiments

This directory contains two new experiments for analyzing layer representations across different fine-tuning methods.

## Experiments

### 1. Layerwise Representation Comparison (`layerwise_representation_comparison.py`)

**Purpose**: Compare layer representations between Full Fine-tuning, LoRA Fine-tuning, and Original models using:
- **Information Imbalance (II)**: Measures how much information is lost when moving from one representation to another
- **Neighbourhood Overlap (NO)**: Computes Jaccard similarity of k-nearest neighbors between representations

**Usage**:
```bash
python paramem/layerwise_representation_comparison.py \
    --representations-dir /home/mmahaut/projects/paramem/hlayer/ \
    --output-dir /home/mmahaut/projects/paramem/output/layerwise_comparison \
    --dataset pile \
    --limit 2500 \
    --k-neighbors 50
```

**Parameters**:
- `--representations-dir`: Directory containing pickled layer representations
- `--output-dir`: Where to save results and plots
- `--dataset`: Dataset name (pile or wikiplus)
- `--limit`: Maximum number of samples to use
- `--k-neighbors`: Number of neighbors for overlap computation

**Expected Input Files**:
The script looks for pickle files in the format:
- `*original*.pickle` or `*base*.pickle`: Original model representations
- `*lora*.pickle`: LoRA fine-tuned model representations
- `*full*.pickle` or `*ft*.pickle`: Full fine-tuning representations

**Outputs**:
1. CSV files with II and NO metrics for each layer pair
2. Heatmaps showing layer-to-layer comparisons
3. Diagonal comparison plots across models

### 2. Layerwise Probing (`layerwise_probing.py`)

**Purpose**: Evaluate how well individual layer representations can predict benchmark performance (MMLU, ARC, etc.) using linear probes.

**Usage**:
```bash
python paramem/layerwise_probing.py \
    --representations-dir /home/mmahaut/projects/paramem/hlayer/ \
    --benchmark-data-dir /home/mmahaut/projects/paramem/benchmark/ \
    --output-dir /home/mmahaut/projects/paramem/output/layerwise_probing \
    --dataset pile \
    --benchmarks mmlu arc hellaswag gsm8k \
    --max-samples 5000
```

**Parameters**:
- `--representations-dir`: Directory with layer representations
- `--benchmark-data-dir`: Directory with benchmark data files
- `--output-dir`: Where to save results
- `--dataset`: Dataset name (pile or wikiplus)
- `--benchmarks`: List of benchmarks to probe
- `--max-samples`: Maximum samples per benchmark (optional)

**Expected Input Files**:
- Representation pickle files (same format as Experiment 1)
- Benchmark data files in TSV format: `id\tlabel\ttext`

**Outputs**:
1. CSV files with probing accuracy per layer per model
2. Line plots showing accuracy across layers
3. Heatmaps comparing models and layers
4. Bar charts showing best layer performance

## Running on SLURM

Example SLURM script for Experiment 1:

```bash
#!/bin/bash
#SBATCH --job-name=layerwise_comparison
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/comparison_%j.out
#SBATCH --error=logs/comparison_%j.err

conda activate paramem

python paramem/layerwise_representation_comparison.py \
    --dataset pile \
    --limit 2500

python paramem/layerwise_representation_comparison.py \
    --dataset wikiplus \
    --limit 2500
```

Example SLURM script for Experiment 2:

```bash
#!/bin/bash
#SBATCH --job-name=layerwise_probing
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/probing_%j.out
#SBATCH --error=logs/probing_%j.err

conda activate paramem

python paramem/layerwise_probing.py \
    --dataset pile \
    --benchmarks mmlu arc hellaswag gsm8k truthful_qa winogrande openbookqa lambada

python paramem/layerwise_probing.py \
    --dataset wikiplus \
    --benchmarks mmlu arc hellaswag gsm8k truthful_qa winogrande openbookqa lambada
```

## Dependencies

Both experiments require:
- `numpy`
- `torch`
- `transformers`
- `dadapy` (for II computation)
- `scikit-learn` (for probing)
- `matplotlib`, `seaborn` (for visualization)
- `pandas`
- `typer`
- `tqdm`

Install with:
```bash
pip install numpy torch transformers dadapy scikit-learn matplotlib seaborn pandas typer tqdm
```

## Output Interpretation

### Experiment 1 (Representation Comparison)

**Information Imbalance (II)**:
- Lower values indicate more similar representations
- Asymmetric: II(A→B) ≠ II(B→A)
- Diagonal values show same-layer similarity across models

**Neighbourhood Overlap (NO)**:
- Range: [0, 1], higher is more similar
- Measures structural similarity of representation spaces
- Symmetric metric

**Key Questions**:
- Which layers change most during fine-tuning?
- Are LoRA and Full FT changes similar or different?
- Do changes propagate through layers?

### Experiment 2 (Probing)

**Probing Accuracy**:
- Shows which layers encode task-relevant information
- Early layers: syntactic features
- Middle layers: semantic features
- Late layers: task-specific features

**Key Questions**:
- Which layers are most informative for each benchmark?
- Does fine-tuning improve task representations in specific layers?
- Are LoRA and Full FT probe patterns similar?

## Notes

1. **Memory Requirements**: Both experiments can be memory-intensive. Adjust `--limit` and `--max-samples` if you encounter OOM errors.

2. **Computation Time**: 
   - Experiment 1: ~2-4 hours for 2500 samples per comparison
   - Experiment 2: ~4-8 hours for all benchmarks

3. **File Organization**: The scripts expect specific naming conventions for representation files. Adjust the file matching logic if your files use different naming.

4. **Extending**: Both scripts are modular. You can easily add:
   - New metrics (CKA, PWCCA, etc.)
   - Different probing architectures (MLP probes, attention probes)
   - Additional benchmarks
   - Fine-tuning checkpoints across training

## Citation

If you use these experiments, please cite the relevant methods:

- Information Imbalance: [DADApy documentation](https://github.com/sissa-data-science/DADApy)
- Linear Probing: Alain & Bengio (2016) "Understanding intermediate layers using linear classifier probes"
