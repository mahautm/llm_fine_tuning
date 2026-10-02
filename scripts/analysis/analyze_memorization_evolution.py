#!/usr/bin/env python3
"""
Analyze memorization evolution across training checkpoints.
Plots training dynamics vs memorization metrics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import typer
from typing import Optional
import numpy as np

from paramem.memorization_metrics import compute_all_memorization_metrics


def analyze_checkpoints(
    model_dir: str,
    data_file: str,
    output_dir: str = "memorization_analysis",
    num_samples: int = 200,
    device: str = "cuda"
):
    """
    Analyze memorization metrics across all checkpoints in a model directory.
    
    Args:
        model_dir: Directory containing checkpoint-* folders
        data_file: Training data CSV file
        output_dir: Where to save results and plots
        num_samples: Number of sequences to test
        device: Device to use for computation
    """
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all checkpoints
    checkpoints = sorted(
        [d for d in model_path.glob("checkpoint-*")],
        key=lambda x: int(x.name.split("-")[1])
    )
    
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {model_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Load training data
    df = pd.read_csv(data_file)
    if "query" in df.columns:
        sequences = df["query"].dropna().tolist()[:num_samples]
    elif "text" in df.columns:
        sequences = df["text"].dropna().tolist()[:num_samples]
    else:
        raise ValueError("Data file must have 'query' or 'text' column")
    
    # Compute metrics for each checkpoint
    results = []
    for checkpoint in checkpoints:
        checkpoint_num = int(checkpoint.name.split("-")[1])
        print(f"\n{'='*60}")
        print(f"Checkpoint {checkpoint_num}")
        print(f"{'='*60}")
        
        try:
            metrics = compute_all_memorization_metrics(
                str(checkpoint), 
                sequences,
                device=device
            )
            
            results.append({
                "checkpoint": checkpoint_num,
                "checkpoint_path": str(checkpoint),
                **metrics
            })
            
            # Save intermediate results
            with open(output_path / f"checkpoint_{checkpoint_num}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_num}: {e}")
            continue
    
    # Save combined results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "memorization_evolution.csv", index=False)
    
    print(f"\n{'='*60}")
    print("All metrics computed. Creating plots...")
    print(f"{'='*60}\n")
    
    # Create plots
    create_memorization_plots(results_df, output_path)
    
    return results_df


def create_memorization_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for memorization metrics."""
    
    sns.set_style("whitegrid")
    
    # 1. All metrics over training
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Memorization Metrics Evolution During Training", fontsize=16, y=1.00)
    
    # Extractability rate
    ax1 = axes[0, 0]
    ax1.plot(df["checkpoint"], df["extractability_rate"], marker='o', linewidth=2)
    ax1.set_xlabel("Training Step (Checkpoint)")
    ax1.set_ylabel("Extractability Rate")
    ax1.set_title("Discoverable Memorization\n(Can sequences be extracted?)")
    ax1.grid(True, alpha=0.3)
    
    # Average tokens recovered
    ax2 = axes[0, 1]
    ax2.plot(df["checkpoint"], df["avg_tokens_recovered"], marker='o', linewidth=2, color='orange')
    ax2.set_xlabel("Training Step (Checkpoint)")
    ax2.set_ylabel("Avg Tokens Recovered")
    ax2.set_title("Token Recovery Rate\n(Partial memorization)")
    ax2.grid(True, alpha=0.3)
    
    # N-gram correlation
    ax3 = axes[1, 0]
    ax3.plot(df["checkpoint"], df["ngram_correlation"], marker='o', linewidth=2, color='green')
    ax3.set_xlabel("Training Step (Checkpoint)")
    ax3.set_ylabel("N-gram Correlation")
    ax3.set_title("Distributional Memorization\n(Correlation with training n-grams)")
    ax3.grid(True, alpha=0.3)
    
    # Intrinsic Dimension
    ax4 = axes[1, 1]
    ax4.plot(df["checkpoint"], df["intrinsic_dimension"], marker='o', linewidth=2, color='red')
    ax4.set_xlabel("Training Step (Checkpoint)")
    ax4.set_ylabel("Intrinsic Dimension")
    ax4.set_title("Embedding Complexity\n(Higher ID = Harder to memorize)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "memorization_evolution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'memorization_evolution.png'}")
    
    # 2. Correlation plot: ID vs Memorization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(
        df["intrinsic_dimension"],
        df["extractability_rate"],
        c=df["checkpoint"],
        s=100,
        cmap='viridis',
        alpha=0.7
    )
    
    # Add trend line
    z = np.polyfit(df["intrinsic_dimension"], df["extractability_rate"], 1)
    p = np.poly1d(z)
    ax.plot(
        df["intrinsic_dimension"], 
        p(df["intrinsic_dimension"]),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}'
    )
    
    # Add correlation coefficient
    corr = np.corrcoef(df["intrinsic_dimension"], df["extractability_rate"])[0, 1]
    ax.text(
        0.05, 0.95,
        f'Correlation: {corr:.3f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_xlabel("Intrinsic Dimension", fontsize=12)
    ax.set_ylabel("Extractability Rate", fontsize=12)
    ax.set_title("Hypothesis: Higher ID → Harder to Memorize", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Step', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "id_vs_memorization.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'id_vs_memorization.png'}")
    
    # 3. Combined memorization score
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Normalize metrics to [0, 1]
    df_norm = df.copy()
    for col in ["extractability_rate", "avg_tokens_recovered", "ngram_correlation"]:
        if df_norm[col].max() - df_norm[col].min() > 0:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    # Plot each metric
    ax.plot(df["checkpoint"], df_norm["extractability_rate"], marker='o', label="Discoverable", linewidth=2)
    ax.plot(df["checkpoint"], df_norm["avg_tokens_recovered"], marker='s', label="Token Recovery", linewidth=2)
    ax.plot(df["checkpoint"], df_norm["ngram_correlation"], marker='^', label="Distributional", linewidth=2)
    
    # Composite score (average of normalized metrics)
    df_norm["composite_memorization"] = df_norm[
        ["extractability_rate", "avg_tokens_recovered", "ngram_correlation"]
    ].mean(axis=1)
    ax.plot(df["checkpoint"], df_norm["composite_memorization"], 
            marker='D', label="Composite Score", linewidth=3, color='black', linestyle='--')
    
    ax.set_xlabel("Training Step (Checkpoint)", fontsize=12)
    ax.set_ylabel("Normalized Memorization Score", fontsize=12)
    ax.set_title("Combined Memorization Metrics (Normalized)", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_memorization.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'combined_memorization.png'}")
    
    plt.close('all')


def main(
    model_dir: str = typer.Argument(..., help="Directory with checkpoint-* folders"),
    data_file: str = typer.Option("data3/wikidata_Mis7.csv", help="Training data CSV"),
    output_dir: str = typer.Option("memorization_analysis", help="Output directory"),
    num_samples: int = typer.Option(200, help="Number of sequences to test"),
    device: str = typer.Option("cuda", help="Device (cuda/cpu)")
):
    """
    Analyze memorization evolution across training checkpoints.
    
    Example:
        srun --ntasks=1 python scripts/analysis/analyze_memorization_evolution.py models2/Llama-3.1-8B-wikiplus-full-ft \\
            --data-file data3/wikidata_Mis7.csv \\
            --output-dir memorization_results
    """
    
    results = analyze_checkpoints(
        model_dir=model_dir,
        data_file=data_file,
        output_dir=output_dir,
        num_samples=num_samples,
        device=device
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(results.to_string())
        print("\n✓ Analysis complete!")
        print(f"  Results saved to: {output_dir}/")


if __name__ == "__main__":
    typer.run(main)
