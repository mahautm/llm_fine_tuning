#!/usr/bin/env python3
"""
Plot layerwise ID and performance evaluation results for 8B LoRA models.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def parse_id_matrixent_log(log_path: Path) -> Tuple[Dict, Dict]:
    """
    Parse intrinsic dimension and entropy from ID evaluation log.
    
    Returns:
        Tuple of (id_dict, entropy_dict) where keys are layer names
    """
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    id_results = {}
    entropy_results = {}
    
    # Parse intrinsic dimension
    id_pattern = re.compile(r"(?P<dataset_name>[\w_]+)_intrinsic_dimension:\s*(?P<layers>(?:\s*layer_\d+:\s*\[.*?\]\s*)+)", re.DOTALL)
    entropy_pattern = re.compile(r"(?P<dataset_name>[\w_]+)_entropy:\s*(?P<layers>(?:\s*layer_\d+:\s*[-+]?\d*\.\d+e[-+]?\d+\s*)+)", re.DOTALL)
    
    for match in id_pattern.finditer(log_content):
        dataset_name = match.group("dataset_name")
        layers = match.group("layers")
        layer_values = re.findall(r"layer_(\d+):\s*\[(.*?)\]", layers)
        
        for layer, values in layer_values:
            layer_key = f"layer_{layer}"
            if layer_key not in id_results:
                id_results[layer_key] = {}
            id_results[layer_key][dataset_name] = [float(v) for v in values.split()]
    
    for match in entropy_pattern.finditer(log_content):
        dataset_name = match.group("dataset_name")
        layers = match.group("layers")
        layer_values = re.findall(r"layer_(\d+):\s*([-+]?\d*\.\d+e[-+]?\d+)", layers)
        
        for layer, value in layer_values:
            layer_key = f"layer_{layer}"
            if layer_key not in entropy_results:
                entropy_results[layer_key] = {}
            entropy_results[layer_key][dataset_name] = float(value)
    
    return id_results, entropy_results


def parse_performance_log(log_path: Path) -> Dict[str, float]:
    """
    Parse benchmark performance metrics from performance evaluation log.
    
    Returns:
        Dictionary mapping metric names to values
    """
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    results = {}
    pattern = re.compile(r"(?P<metric>[\w_]+):\s*(?P<value>[-+]?\d*\.\d+|\d+)", re.MULTILINE)
    
    for match in pattern.finditer(log_content):
        metric = match.group("metric")
        value = match.group("value")
        results[metric] = float(value)
    
    return results


def plot_layerwise_id(id_data: Dict[str, Dict[str, List[float]]], model_name: str, output_path: Path):
    """
    Plot layerwise intrinsic dimension across multiple scales.
    
    Args:
        id_data: Dictionary mapping layer names to dataset->ID_values
        model_name: Name of the model (for title)
        output_path: Where to save the plot
    """
    # Prepare data for plotting
    layers = sorted([int(k.split('_')[1]) for k in id_data.keys()])
    
    # Get dataset names from first layer
    first_layer = f"layer_{layers[0]}"
    datasets = list(id_data[first_layer].keys())
    
    # Create subplots: one for each dataset
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 5*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Get number of scales (usually 5)
        n_scales = len(id_data[first_layer][dataset])
        
        # Plot each scale
        for scale_idx in range(n_scales):
            scale_values = []
            for layer in layers:
                layer_key = f"layer_{layer}"
                if layer_key in id_data and dataset in id_data[layer_key]:
                    scale_values.append(id_data[layer_key][dataset][scale_idx])
                else:
                    scale_values.append(np.nan)
            
            scale_label = f"Scale {2**(scale_idx+1)}"
            ax.plot(layers, scale_values, marker='o', label=scale_label, linewidth=2, markersize=4)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Intrinsic Dimension', fontsize=12)
        ax.set_title(f'{model_name} - {dataset} - Intrinsic Dimension by Layer', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved layerwise ID plot: {output_path}")


def plot_layerwise_entropy(entropy_data: Dict[str, Dict[str, float]], model_name: str, output_path: Path):
    """
    Plot layerwise entropy.
    
    Args:
        entropy_data: Dictionary mapping layer names to dataset->entropy_value
        model_name: Name of the model (for title)
        output_path: Where to save the plot
    """
    layers = sorted([int(k.split('_')[1]) for k in entropy_data.keys()])
    
    # Get dataset names
    first_layer = f"layer_{layers[0]}"
    datasets = list(entropy_data[first_layer].keys())
    
    plt.figure(figsize=(14, 6))
    
    for dataset in datasets:
        entropy_values = []
        for layer in layers:
            layer_key = f"layer_{layer}"
            if layer_key in entropy_data and dataset in entropy_data[layer_key]:
                entropy_values.append(entropy_data[layer_key][dataset])
            else:
                entropy_values.append(np.nan)
        
        plt.plot(layers, entropy_values, marker='o', label=dataset, linewidth=2, markersize=6)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Matrix Entropy', fontsize=12)
    plt.title(f'{model_name} - Matrix Entropy by Layer', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Entropy values are often very small
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved layerwise entropy plot: {output_path}")


def plot_benchmark_comparison(perf_data: Dict[str, Dict[str, float]], output_path: Path):
    """
    Plot benchmark performance comparison between models.
    
    Args:
        perf_data: Dictionary mapping model_name to metric->value dict
        output_path: Where to save the plot
    """
    # Convert to DataFrame
    df = pd.DataFrame(perf_data).T
    
    # Filter to relevant accuracy metrics
    accuracy_cols = [col for col in df.columns if 'accuracy' in col or 'nwp' in col]
    df_acc = df[accuracy_cols]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(accuracy_cols))
    width = 0.35
    
    model_names = list(perf_data.keys())
    colors = sns.color_palette("husl", len(model_names))
    
    for idx, model_name in enumerate(model_names):
        values = df_acc.loc[model_name].values
        offset = width * (idx - len(model_names)/2 + 0.5)
        ax.bar(x + offset, values, width, label=model_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Accuracy / Score', fontsize=12)
    ax.set_title('Benchmark Performance Comparison - 8B LoRA Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([col.replace('_', ' ').title() for col in accuracy_cols], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved benchmark comparison plot: {output_path}")


def main():
    """Main function to generate all plots for 8B LoRA results."""
    
    # Define checkpoint paths
    checkpoints = {
        "Llama-3.1-8B LoRA (Wikiplus)": Path("/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-1540"),
        "Llama-3.1-8B LoRA (Pile)": Path("/home/mmahaut/projects/paramem/models2/Llama-3.1-8B-Instruct-fsdp-1-pile-lr1e-4/checkpoint-1540"),
    }
    
    output_dir = Path("/home/mmahaut/projects/paramem/plots_8b_lora")
    output_dir.mkdir(exist_ok=True)
    
    all_perf_data = {}
    
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n📊 Processing {model_name}...")
        
        slurm_logs = checkpoint_path / "slurm_logs"
        if not slurm_logs.exists():
            print(f"⚠️  No slurm_logs directory found for {model_name}")
            continue
        
        # Parse ID and entropy
        id_log = slurm_logs / "ID_matrixent.out"
        if id_log.exists():
            id_data, entropy_data = parse_id_matrixent_log(id_log)
            
            # Plot layerwise ID
            safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            id_output = output_dir / f"{safe_name}_layerwise_ID.png"
            plot_layerwise_id(id_data, model_name, id_output)
            
            # Plot layerwise entropy
            entropy_output = output_dir / f"{safe_name}_layerwise_entropy.png"
            plot_layerwise_entropy(entropy_data, model_name, entropy_output)
        else:
            print(f"⚠️  No ID_matrixent.out found for {model_name}")
        
        # Parse performance
        perf_log = slurm_logs / "performance_evaluation.out"
        if perf_log.exists():
            perf_data = parse_performance_log(perf_log)
            all_perf_data[model_name] = perf_data
        else:
            print(f"⚠️  No performance_evaluation.out found for {model_name}")
    
    # Plot benchmark comparison across all models
    if all_perf_data:
        comparison_output = output_dir / "benchmark_comparison_8b_lora.png"
        plot_benchmark_comparison(all_perf_data, comparison_output)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
