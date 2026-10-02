#!/usr/bin/env python3
"""
Correlate most oscillating layer's ID with memorization metrics.
"""

import json
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_layer_id(layer_num=31):
    """Extract ID for a specific layer at each checkpoint."""
    id_files = sorted(glob.glob("models3/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-*/slurm_logs/ID_matrixent.out"))
    
    layer_data = {}
    
    for f in id_files:
        ckpt = int(os.path.basename(os.path.dirname(os.path.dirname(f))).replace("checkpoint-", ""))
        with open(f) as fh:
            content = fh.read()
            # Look for pile_19_short_train intrinsic_dimension
            pattern = r"pile_19_short_train_intrinsic_dimension:\s*\n((?:\s+layer_\d+:.*\n)*)"
            matches = re.findall(pattern, content)
            
            if matches:
                layer_lines = re.findall(r"layer_(\d+):\s*\[([\d\.\s]+)\]", matches[0])
                
                for ln, values in layer_lines:
                    if int(ln) == layer_num:
                        values_clean = re.sub(r"\s+", " ", values.strip())
                        if values_clean:
                            try:
                                values_array = [float(x) for x in values_clean.split()]
                                if values_array:
                                    layer_data[ckpt] = values_array[-1]
                            except ValueError:
                                continue
                        break
    
    return layer_data

def extract_models4_memorization():
    """Extract memorization metrics from models4 v3 LoRA."""
    mem_files = sorted(glob.glob("models4/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4-v3/checkpoint-*/slurm_logs/memorization_metrics.json"))
    mem_data = {}
    
    for f in mem_files:
        ckpt = int(os.path.basename(os.path.dirname(os.path.dirname(f))).replace("checkpoint-", ""))
        with open(f) as fh:
            m = json.load(fh)
            mem_data[ckpt] = {
                'extractability_rate': m.get('extractability_rate', 0),
                'avg_tokens_recovered': m.get('avg_tokens_recovered', 0),
                'ngram_correlation': m.get('ngram_correlation', 0),
                'train_nll': m.get('train_nll', 0),
                'train_perplexity': m.get('train_perplexity', 0),
                'exposure_delta': m.get('exposure_delta', 0),
                'exposure_positive_rate': m.get('exposure_positive_rate', 0),
            }
    
    return mem_data

def plot_layer_evolution(layer_data, layer_num):
    """Plot single layer ID evolution."""
    steps = sorted(layer_data.keys())
    ids = [layer_data[s] for s in steps]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(steps, ids, 'o-', linewidth=3, markersize=10, color='darkblue')
    
    # Mark direction changes
    for i in range(1, len(ids)-1):
        if (ids[i] > ids[i-1] and ids[i] > ids[i+1]) or \
           (ids[i] < ids[i-1] and ids[i] < ids[i+1]):
            ax.axvline(steps[i], color='red', alpha=0.3, linestyle='--', linewidth=2)
            ax.plot(steps[i], ids[i], 'ro', markersize=12, alpha=0.5)
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intrinsic Dimension', fontsize=14, fontweight='bold')
    ax.set_title(f'Layer {layer_num} Intrinsic Dimension Evolution\n(Most Variable Layer - LoRA Wikiplus)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add variance annotation
    var = np.var(ids)
    ax.text(0.02, 0.98, f'Variance: {var:.2f}\nRange: [{min(ids):.2f}, {max(ids):.2f}]',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = f"plots_memorization/layer_{layer_num}_id_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Layer evolution plot saved to {output_path}")

def plot_correlations(layer_data, mem_data, layer_num):
    """Plot correlations between layer ID and memorization metrics."""
    # Find common checkpoints
    common_steps = sorted(set(layer_data.keys()) & set(mem_data.keys()))
    
    if not common_steps:
        print("❌ No common checkpoints found!")
        return
    
    print(f"✅ Found {len(common_steps)} common checkpoints")
    
    # Extract values
    steps = common_steps
    ids = [layer_data[s] for s in steps]
    extractability = [mem_data[s]['extractability_rate'] for s in steps]
    avg_tokens = [mem_data[s]['avg_tokens_recovered'] for s in steps]
    ngram = [mem_data[s]['ngram_correlation'] for s in steps]
    nll = [mem_data[s]['train_nll'] for s in steps]
    perplexity = [mem_data[s]['train_perplexity'] for s in steps]
    exposure = [mem_data[s]['exposure_delta'] for s in steps]
    exposure_pos = [mem_data[s]['exposure_positive_rate'] for s in steps]
    
    # Create 3x2 plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(f'Layer {layer_num} ID vs Memorization Metrics (LoRA Wikiplus)\nModels3 Layer ID vs Models4 Memorization', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('Extractability Rate', extractability, 'tab:red'),
        ('Avg Tokens Recovered', avg_tokens, 'tab:orange'),
        ('N-gram Correlation', ngram, 'tab:green'),
        ('Train NLL', nll, 'tab:blue'),
        ('Train Perplexity', perplexity, 'tab:purple'),
        ('Exposure Delta', exposure, 'tab:brown'),
    ]
    
    for idx, (metric_name, metric_vals, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Single axis with both metrics (normalized)
        ax.set_xlabel('Training Steps', fontsize=11)
        
        # Normalize both to 0-1 for better visualization
        ids_norm = np.array(ids)
        ids_norm = (ids_norm - ids_norm.min()) / (ids_norm.max() - ids_norm.min() + 1e-10)
        
        metric_vals_norm = np.array(metric_vals)
        if metric_vals_norm.max() - metric_vals_norm.min() > 1e-10:
            metric_vals_norm = (metric_vals_norm - metric_vals_norm.min()) / (metric_vals_norm.max() - metric_vals_norm.min())
        
        # Plot both on same scale
        ax.plot(steps, ids_norm, 'o-', color='#2E86AB', linewidth=2.5, markersize=8, 
                label=f'Layer {layer_num} ID (norm)', alpha=0.8)
        ax.plot(steps, metric_vals_norm, 's-', color=color, linewidth=2.5, markersize=8, 
                label=f'{metric_name} (norm)', alpha=0.8)
        
        ax.set_ylabel('Normalized Value (0-1)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        
        # Calculate correlation
        corr = np.corrcoef(ids, metric_vals)[0, 1]
        ax.set_title(f'Layer {layer_num} ID vs {metric_name}\nCorrelation: r = {corr:.3f}', 
                     fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = f"plots_memorization/layer_{layer_num}_memorization_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation plot saved to {output_path}")
    
    # Print correlation summary
    print(f"\n=== Layer {layer_num} Correlation Summary ===")
    for metric_name, metric_vals, _ in metrics:
        corr = np.corrcoef(ids, metric_vals)[0, 1]
        print(f"{metric_name:25s}: r = {corr:+.3f}")
    
    # Also print exposure_positive_rate correlation
    corr_exp_pos = np.corrcoef(ids, exposure_pos)[0, 1]
    print(f"{'Exposure Positive Rate':25s}: r = {corr_exp_pos:+.3f}")
    
    # Print statistics
    print(f"\n=== Layer {layer_num} Statistics ===")
    print(f"ID range: [{min(ids):.2f}, {max(ids):.2f}]")
    print(f"ID variance: {np.var(ids):.2f}")
    print(f"Direction changes: {sum(1 for i in range(1, len(ids)-1) if (ids[i] > ids[i-1]) != (ids[i+1] > ids[i]))}")

def analyze_multiple_layers():
    """Compare top oscillating layers."""
    layers_to_check = [26, 27, 28, 29, 30, 31]  # Most oscillating layers
    
    mem_data = extract_models4_memorization()
    
    print("\n=== Comparing Multiple Oscillating Layers ===")
    
    all_correlations = {}
    
    for layer_num in layers_to_check:
        layer_data = extract_layer_id(layer_num)
        common_steps = sorted(set(layer_data.keys()) & set(mem_data.keys()))
        
        if not common_steps:
            continue
        
        ids = [layer_data[s] for s in common_steps]
        extractability = [mem_data[s]['extractability_rate'] for s in common_steps]
        nll = [mem_data[s]['train_nll'] for s in common_steps]
        
        corr_extr = np.corrcoef(ids, extractability)[0, 1]
        corr_nll = np.corrcoef(ids, nll)[0, 1]
        var = np.var(ids)
        
        all_correlations[layer_num] = {
            'extractability': corr_extr,
            'nll': corr_nll,
            'variance': var
        }
        
        print(f"Layer {layer_num}: var={var:.2f}, r(extr)={corr_extr:+.3f}, r(nll)={corr_nll:+.3f}")
    
    # Find best layer for each metric
    best_extr = max(all_correlations.items(), key=lambda x: abs(x[1]['extractability']))
    best_nll = max(all_correlations.items(), key=lambda x: abs(x[1]['nll']))
    
    print(f"\nStrongest correlations:")
    print(f"  Extractability: Layer {best_extr[0]} (r={best_extr[1]['extractability']:+.3f})")
    print(f"  Train NLL: Layer {best_nll[0]} (r={best_nll[1]['nll']:+.3f})")

if __name__ == "__main__":
    # Use layer 31 (highest variance) or could use 26-29 (most direction changes)
    layer_num = 31
    
    print(f"Analyzing Layer {layer_num} (highest variance)...")
    layer_data = extract_layer_id(layer_num)
    
    print(f"Extracting memorization data...")
    mem_data = extract_models4_memorization()
    
    print(f"\n1. Plotting Layer {layer_num} ID evolution...")
    plot_layer_evolution(layer_data, layer_num)
    
    print(f"\n2. Plotting correlations with memorization...")
    plot_correlations(layer_data, mem_data, layer_num)
    
    print(f"\n3. Comparing multiple oscillating layers...")
    analyze_multiple_layers()
    
    print("\n✅ All analysis complete!")
