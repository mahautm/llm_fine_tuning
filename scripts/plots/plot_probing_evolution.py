#!/usr/bin/env python3
"""
Aggregate and plot layerwise probing results across all checkpoints.
Creates evolution plots showing how probing performance changes during training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import re


def collect_probing_results(model_dir: str, benchmark: str = "mmlu") -> Dict[int, pd.DataFrame]:
    """
    Collect probing results from all checkpoints.
    
    Returns:
        Dictionary mapping checkpoint step to probing results DataFrame
    """
    model_path = Path(model_dir)
    results = {}
    
    print(f"🔍 Scanning {model_path} for probing results...")
    
    for checkpoint_dir in sorted(model_path.glob("checkpoint-*")):
        # Extract checkpoint step number
        match = re.search(r'checkpoint-(\d+)', checkpoint_dir.name)
        if not match:
            continue
        
        step = int(match.group(1))
        
        # Look for probing results
        probing_file = checkpoint_dir / "slurm_logs" / f"probing_{benchmark}.csv"
        
        if probing_file.exists():
            try:
                df = pd.read_csv(probing_file)
                results[step] = df
                print(f"  ✓ Step {step}: {len(df)} layers")
            except Exception as e:
                print(f"  ✗ Step {step}: Error reading file - {e}")
    
    print(f"✅ Found probing results for {len(results)} checkpoints")
    return results


def plot_probing_evolution(
    results: Dict[int, pd.DataFrame],
    model_name: str,
    benchmark: str,
    output_path: Path,
    layer_selection: str = "all"  # "all", "best", or specific layer numbers
):
    """
    Plot how probing performance evolves across checkpoints.
    
    Args:
        results: Dictionary mapping checkpoint step to results DataFrame
        model_name: Name of the model for title
        benchmark: Benchmark name
        output_path: Where to save the plot
        layer_selection: Which layers to plot ("all", "best", or comma-separated layer indices)
    """
    if not results:
        print("⚠️  No results to plot")
        return
    
    steps = sorted(results.keys())
    
    if layer_selection == "all":
        # Plot all layers
        first_df = results[steps[0]]
        layers_to_plot = sorted(first_df['layer'].unique())
    elif layer_selection == "best":
        # Plot only the best layer at each checkpoint
        layers_to_plot = ["best"]
    else:
        # Specific layers
        layers_to_plot = [int(l) for l in layer_selection.split(',')]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Accuracy evolution
    if layer_selection == "best":
        best_accuracies = []
        best_layers = []
        
        for step in steps:
            df = results[step]
            best_idx = df['accuracy'].idxmax()
            best_row = df.iloc[best_idx]
            best_accuracies.append(best_row['accuracy'])
            best_layers.append(int(best_row['layer']))
        
        ax1.plot(steps, best_accuracies, marker='o', linewidth=2, markersize=8, 
                label='Best Layer', color='darkblue')
        ax1.set_ylabel('Accuracy', fontsize=14)
        
        # Plot 2: Which layer was best
        ax2.plot(steps, best_layers, marker='s', linewidth=2, markersize=8,
                color='darkgreen')
        ax2.set_ylabel('Best Layer Index', fontsize=14)
        ax2.set_title(f'Best Performing Layer Over Training - {benchmark.upper()}', 
                     fontsize=16, fontweight='bold')
    else:
        # Plot multiple layers
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))
        
        for idx, layer in enumerate(layers_to_plot):
            accuracies = []
            cv_means = []
            cv_stds = []
            
            for step in steps:
                df = results[step]
                layer_data = df[df['layer'] == layer]
                
                if not layer_data.empty:
                    row = layer_data.iloc[0]
                    accuracies.append(row['accuracy'])
                    cv_means.append(row['cv_mean'])
                    cv_stds.append(row['cv_std'])
                else:
                    accuracies.append(np.nan)
                    cv_means.append(np.nan)
                    cv_stds.append(np.nan)
            
            # Plot test accuracy
            ax1.plot(steps, accuracies, marker='o', linewidth=2, markersize=6,
                    label=f'Layer {layer}', color=colors[idx])
            
            # Plot CV mean with std bands
            ax2.plot(steps, cv_means, marker='o', linewidth=2, markersize=6,
                    label=f'Layer {layer}', color=colors[idx])
            ax2.fill_between(steps, 
                            np.array(cv_means) - np.array(cv_stds),
                            np.array(cv_means) + np.array(cv_stds),
                            alpha=0.2, color=colors[idx])
        
        ax1.set_ylabel('Test Accuracy', fontsize=14)
        ax2.set_ylabel('CV Mean Accuracy', fontsize=14)
        ax2.set_title(f'Cross-Validation Performance - {benchmark.upper()}', 
                     fontsize=16, fontweight='bold')
    
    # Common formatting
    ax1.set_xlabel('Training Step', fontsize=14)
    ax1.set_title(f'{model_name} - Probing Performance Evolution - {benchmark.upper()}', 
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel('Training Step', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if layer_selection != "best":
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved evolution plot: {output_path}")


def plot_probing_heatmap(
    results: Dict[int, pd.DataFrame],
    model_name: str,
    benchmark: str,
    output_path: Path
):
    """
    Create a heatmap showing probing accuracy for all layers across checkpoints.
    """
    if not results:
        print("⚠️  No results to plot")
        return
    
    steps = sorted(results.keys())
    
    # Get all layer indices
    all_layers = set()
    for df in results.values():
        all_layers.update(df['layer'].values)
    all_layers = sorted(all_layers)
    
    # Build matrix: rows = layers, columns = steps
    matrix = np.full((len(all_layers), len(steps)), np.nan)
    
    for step_idx, step in enumerate(steps):
        df = results[step]
        for layer_idx, layer in enumerate(all_layers):
            layer_data = df[df['layer'] == layer]
            if not layer_data.empty:
                matrix[layer_idx, step_idx] = layer_data.iloc[0]['accuracy']
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.set_yticks(range(len(all_layers)))
    ax.set_yticklabels([f"Layer {l}" for l in all_layers])
    
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title(f'{model_name} - Probing Accuracy Heatmap - {benchmark.upper()}', 
                fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved heatmap: {output_path}")


def main():
    """Main function to aggregate and plot probing results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot layerwise probing evolution")
    parser.add_argument("--model-dir", "-m", required=True,
                       help="Model directory containing checkpoints")
    parser.add_argument("--benchmark", "-b", default="mmlu",
                       help="Benchmark name (default: mmlu)")
    parser.add_argument("--output-dir", "-o", default=None,
                       help="Output directory (default: model_dir/probing_plots)")
    parser.add_argument("--layers", "-l", default="best",
                       help="Layers to plot: 'all', 'best', or comma-separated indices (default: best)")
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = Path(args.model_dir)
    model_name = model_path.name
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path / "probing_plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📊 Plotting Layerwise Probing Evolution")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Output: {output_dir}")
    
    # Collect results
    results = collect_probing_results(args.model_dir, args.benchmark)
    
    if not results:
        print("❌ No probing results found!")
        return
    
    # Plot evolution
    print(f"\n📈 Creating evolution plots...")
    plot_probing_evolution(
        results,
        model_name,
        args.benchmark,
        output_dir / f"probing_evolution_{args.benchmark}.png",
        layer_selection=args.layers
    )
    
    # Plot heatmap
    print(f"\n🔥 Creating heatmap...")
    plot_probing_heatmap(
        results,
        model_name,
        args.benchmark,
        output_dir / f"probing_heatmap_{args.benchmark}.png"
    )
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
