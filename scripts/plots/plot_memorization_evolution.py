#!/usr/bin/env python3
"""
Plot memorization metrics evolution across training checkpoints.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List

NEW_METRICS = [
    "train_nll",
    "train_perplexity",
    "exposure_delta",
    "exposure_positive_rate",
]

def collect_memorization_metrics(model_dir: Path) -> pd.DataFrame:
    """
    Collect memorization metrics from all checkpoints in a model directory.
    
    Args:
        model_dir: Path to model directory containing checkpoints
        
    Returns:
        DataFrame with columns: checkpoint, step, extractability_rate, 
                                avg_tokens_recovered, ngram_correlation, intrinsic_dimension
    """
    data = []
    
    # Find all checkpoint directories
    checkpoints = sorted(model_dir.glob("checkpoint-*"), 
                         key=lambda x: int(x.name.split('-')[1]))
    
    for ckpt_dir in checkpoints:
        metrics_file = ckpt_dir / "slurm_logs" / "memorization_metrics.json"
        
        if not metrics_file.exists():
            continue
            
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Skip empty metrics
            if not metrics or metrics == {}:
                continue
                
            step = int(ckpt_dir.name.split('-')[1])
            
            row = {
                'checkpoint': ckpt_dir.name,
                'step': step,
                'extractability_rate': metrics.get('extractability_rate', 0.0),
                'avg_tokens_recovered': metrics.get('avg_tokens_recovered', 0.0),
                'ngram_correlation': metrics.get('ngram_correlation', 0.0),
                'intrinsic_dimension': metrics.get('intrinsic_dimension', 0.0),
                'num_sequences_tested': metrics.get('num_sequences_tested', 0)
            }
            for key in NEW_METRICS:
                row[key] = metrics.get(key, None)

            data.append(row)
        except Exception as e:
            print(f"⚠️  Error reading {metrics_file}: {e}")
            continue
    
    if not data:
        return pd.DataFrame()
        
    return pd.DataFrame(data)


def plot_memorization_evolution(df: pd.DataFrame, model_name: str, output_path: Path):
    """
    Plot memorization metrics evolution across training steps.
    
    Args:
        df: DataFrame with memorization metrics
        model_name: Name of the model for title
        output_path: Where to save the plot
    """
    if df.empty:
        print(f"⚠️  No data to plot for {model_name}")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(f'{model_name} - Memorization Metrics Evolution', fontsize=16, fontweight='bold')
    
    # Plot 1: Extractability Rate
    ax = axes[0, 0]
    ax.plot(df['step'], df['extractability_rate'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Extractability Rate', fontsize=12)
    ax.set_title('Discoverable Memorization (Extractability Rate)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 2: Average Tokens Recovered
    ax = axes[0, 1]
    ax.plot(df['step'], df['avg_tokens_recovered'], marker='o', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Avg Tokens Recovered', fontsize=12)
    ax.set_title('Average Tokens Recovered per Sequence', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 3: N-gram Correlation
    ax = axes[1, 0]
    ax.plot(df['step'], df['ngram_correlation'], marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('N-gram Correlation', fontsize=12)
    ax.set_title('Distributional Memorization (N-gram Correlation)', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Intrinsic Dimension
    ax = axes[1, 1]
    ax.plot(df['step'], df['intrinsic_dimension'], marker='o', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Intrinsic Dimension', fontsize=12)
    ax.set_title('Embedding Intrinsic Dimension', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 5: Train NLL / Perplexity
    ax = axes[2, 0]
    if 'train_nll' in df.columns and df['train_nll'].notna().any():
        ax.plot(df['step'], df['train_nll'], marker='o', linewidth=2, markersize=8, color='purple', label='Train NLL')
        ax.set_ylabel('Train NLL', fontsize=12)
        ax.set_title('Train NLL', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        if 'train_perplexity' in df.columns and df['train_perplexity'].notna().any():
            ax2.plot(df['step'], df['train_perplexity'], marker='s', linewidth=1.5, markersize=6, color='gray', label='Train PPL')
            ax2.set_ylabel('Train Perplexity', fontsize=12)
        ax.legend(loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No NLL data', ha='center', va='center')
        ax.axis('off')

    # Plot 6: Exposure delta / positive rate
    ax = axes[2, 1]
    if 'exposure_delta' in df.columns and df['exposure_delta'].notna().any():
        ax.plot(df['step'], df['exposure_delta'], marker='o', linewidth=2, markersize=8, color='brown', label='Exposure Δ')
        ax.set_ylabel('Exposure Δ', fontsize=12)
        ax.set_title('Exposure', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        if 'exposure_positive_rate' in df.columns and df['exposure_positive_rate'].notna().any():
            ax2.plot(df['step'], df['exposure_positive_rate'], marker='s', linewidth=1.5, markersize=6, color='teal', label='% positive')
            ax2.set_ylabel('Positive rate', fontsize=12)
        ax.legend(loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No exposure data', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    plt.close()


def main():
    """Main function to process all models."""
    sns.set_style("whitegrid")
    
    models_dir = Path("/home/mmahaut/projects/paramem/models4")
    output_dir = Path("/home/mmahaut/projects/paramem/plots_memorization")
    output_dir.mkdir(exist_ok=True)
    
    # Process all model directories
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
            
        print(f"\n📊 Processing {model_dir.name}...")
        
        # Collect metrics
        df = collect_memorization_metrics(model_dir)
        
        if df.empty:
            print(f"⚠️  No valid memorization metrics found")
            continue
        
        print(f"✓ Found {len(df)} checkpoints with metrics")
        print(f"  Steps: {df['step'].min()} → {df['step'].max()}")
        
        # Generate plot
        output_path = output_dir / f"{model_dir.name}_memorization_evolution.png"
        plot_memorization_evolution(df, model_dir.name, output_path)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
