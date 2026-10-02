#!/usr/bin/env python3
"""
Analyze layer-wise intrinsic dimension evolution to find where ID reverses direction.
"""

import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8')

def extract_layerwise_id():
    """Extract ID for each layer at each checkpoint."""
    id_files = sorted(glob.glob("models3/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4/checkpoint-*/slurm_logs/ID_matrixent.out"))
    
    data = {}  # {checkpoint: {layer: id_value}}
    
    for f in id_files:
        ckpt = int(os.path.basename(os.path.dirname(os.path.dirname(f))).replace("checkpoint-", ""))
        with open(f) as fh:
            content = fh.read()
            # Look for pile_19_short_train intrinsic_dimension
            pattern = r"pile_19_short_train_intrinsic_dimension:\s*\n((?:\s+layer_\d+:.*\n)*)"
            matches = re.findall(pattern, content)
            
            if matches:
                layer_data = matches[0]
                layer_lines = re.findall(r"layer_(\d+):\s*\[([\d\.\s]+)\]", layer_data)
                
                data[ckpt] = {}
                for layer_num, values in layer_lines:
                    values_clean = re.sub(r"\s+", " ", values.strip())
                    if values_clean:
                        try:
                            values_array = [float(x) for x in values_clean.split()]
                            # Take the last value (highest alpha)
                            if values_array:
                                data[ckpt][int(layer_num)] = values_array[-1]
                        except ValueError:
                            continue
    
    return data

def analyze_layer_patterns(data):
    """Analyze which layers show interesting patterns."""
    checkpoints = sorted(data.keys())
    all_layers = sorted(set(layer for ckpt_data in data.values() for layer in ckpt_data.keys()))
    
    print(f"Analyzing {len(checkpoints)} checkpoints with {len(all_layers)} layers")
    print(f"Checkpoints: {checkpoints}")
    
    # Calculate average ID at each checkpoint
    avg_ids = []
    for ckpt in checkpoints:
        layer_vals = list(data[ckpt].values())
        avg_ids.append(np.mean(layer_vals))
    
    print("\n=== Average ID Evolution ===")
    for i, ckpt in enumerate(checkpoints):
        direction = ""
        if i > 0:
            change = avg_ids[i] - avg_ids[i-1]
            direction = f" ({'+' if change > 0 else ''}{change:.2f})"
        print(f"Step {ckpt:4d}: {avg_ids[i]:6.2f}{direction}")
    
    # Find layers with the most variance
    layer_variances = {}
    for layer in all_layers:
        vals = [data[ckpt][layer] for ckpt in checkpoints if layer in data[ckpt]]
        if vals:
            layer_variances[layer] = np.var(vals)
    
    # Sort by variance
    sorted_layers = sorted(layer_variances.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== Top 10 Layers by Variance ===")
    for layer, var in sorted_layers[:10]:
        vals = [data[ckpt][layer] for ckpt in checkpoints if layer in data[ckpt]]
        print(f"Layer {layer:2d}: variance={var:.2f}, range=[{min(vals):.2f}, {max(vals):.2f}]")
    
    # Find layers that show "up then down" pattern at key transition points
    print("\n=== Layers with Direction Changes ===")
    for layer in all_layers:
        vals = [data[ckpt][layer] for ckpt in checkpoints if layer in data[ckpt]]
        if len(vals) < 4:
            continue
        
        # Count direction changes
        directions = []
        for i in range(1, len(vals)):
            directions.append(1 if vals[i] > vals[i-1] else -1)
        
        # Count sign changes
        sign_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
        
        if sign_changes >= 3:  # Multiple direction changes
            print(f"Layer {layer:2d}: {sign_changes} direction changes, values: {[f'{v:.1f}' for v in vals[:5]]}...")
    
    return data, checkpoints, all_layers, sorted_layers

def plot_layerwise_evolution(data, checkpoints, all_layers, sorted_layers):
    """Plot layer-wise ID evolution."""
    # Plot top varying layers
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Layer-wise Intrinsic Dimension Evolution - LoRA Wikiplus\n(Top 9 Most Variable Layers)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (layer, var) in enumerate(sorted_layers[:9]):
        ax = axes[idx // 3, idx % 3]
        vals = [data[ckpt][layer] for ckpt in checkpoints if layer in data[ckpt]]
        
        ax.plot(checkpoints, vals, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Intrinsic Dimension')
        ax.set_title(f'Layer {layer} (var={var:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Mark direction changes
        for i in range(1, len(vals)-1):
            if (vals[i] > vals[i-1] and vals[i] > vals[i+1]) or \
               (vals[i] < vals[i-1] and vals[i] < vals[i+1]):
                ax.axvline(checkpoints[i], color='red', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = "plots_memorization/layerwise_id_evolution_top9.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {output_path}")
    
    # Plot all layers in a heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Create matrix: layers x checkpoints
    matrix = np.zeros((len(all_layers), len(checkpoints)))
    for i, layer in enumerate(all_layers):
        for j, ckpt in enumerate(checkpoints):
            if layer in data[ckpt]:
                matrix[i, j] = data[ckpt][layer]
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax.set_xlabel('Checkpoint Step')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels(checkpoints, rotation=45)
    ax.set_yticks(range(0, len(all_layers), 4))
    ax.set_yticklabels(range(0, len(all_layers), 4))
    ax.set_title('Layer-wise Intrinsic Dimension Heatmap - LoRA Wikiplus', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intrinsic Dimension', rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_path = "plots_memorization/layerwise_id_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to {output_path}")
    
    # Plot average with layer groups
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Group layers: early (0-10), middle (11-21), late (22-32)
    early_layers = list(range(0, 11))
    middle_layers = list(range(11, 22))
    late_layers = list(range(22, 33))
    
    avg_ids = []
    early_ids = []
    middle_ids = []
    late_ids = []
    
    for ckpt in checkpoints:
        all_vals = list(data[ckpt].values())
        avg_ids.append(np.mean(all_vals))
        
        early_vals = [data[ckpt][l] for l in early_layers if l in data[ckpt]]
        middle_vals = [data[ckpt][l] for l in middle_layers if l in data[ckpt]]
        late_vals = [data[ckpt][l] for l in late_layers if l in data[ckpt]]
        
        early_ids.append(np.mean(early_vals) if early_vals else 0)
        middle_ids.append(np.mean(middle_vals) if middle_vals else 0)
        late_ids.append(np.mean(late_vals) if late_vals else 0)
    
    ax.plot(checkpoints, avg_ids, 'o-', linewidth=3, markersize=8, label='All Layers (avg)', color='black')
    ax.plot(checkpoints, early_ids, 's-', linewidth=2, markersize=6, label='Early Layers (0-10)', alpha=0.7)
    ax.plot(checkpoints, middle_ids, '^-', linewidth=2, markersize=6, label='Middle Layers (11-21)', alpha=0.7)
    ax.plot(checkpoints, late_ids, 'd-', linewidth=2, markersize=6, label='Late Layers (22-32)', alpha=0.7)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Average Intrinsic Dimension', fontsize=12)
    ax.set_title('Layer Group ID Evolution - LoRA Wikiplus', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "plots_memorization/layerwise_id_groups.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Layer groups plot saved to {output_path}")

if __name__ == "__main__":
    print("Extracting layer-wise ID data...")
    data = extract_layerwise_id()
    
    print("\nAnalyzing patterns...")
    data, checkpoints, all_layers, sorted_layers = analyze_layer_patterns(data)
    
    print("\nGenerating plots...")
    plot_layerwise_evolution(data, checkpoints, all_layers, sorted_layers)
    
    print("\n✅ Analysis complete!")
