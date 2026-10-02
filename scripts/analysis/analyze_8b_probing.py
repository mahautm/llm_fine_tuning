#!/usr/bin/env python3
"""
Analysis script for layerwise probing results in Llama-3.1-8B models3 directory.
Creates plots showing probing accuracy evolution across checkpoints and layers.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_probing_csv(csv_path):
    """Parse layerwise probing CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def collect_probing_data():
    """Collect all probing data from models3."""
    base_path = "/home/mmahaut/projects/paramem/models3"
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 Scanning models3 for layerwise probing results...")
    
    # Iterate through model directories
    for model_dir in os.listdir(base_path):
        if not model_dir.startswith("Llama-3.1-8B-Instruct-fsdp"):
            continue
            
        model_path = os.path.join(base_path, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        # Parse configuration from directory name
        if 'fsdp-1' in model_dir:
            training_type = 'LoRA'
        elif 'fsdp-0' in model_dir:
            training_type = 'Full-FT'
        else:
            continue
        
        if 'pile' in model_dir.lower():
            dataset = 'pile'
        elif 'mmlu' in model_dir.lower():
            dataset = 'mmlu'
        elif 'wiki' in model_dir.lower():
            dataset = 'wikiplus'
        else:
            dataset = 'unknown'
        
        model_key = f"{training_type}_{dataset}"
        
        # Find checkpoints
        checkpoint_dirs = [d for d in os.listdir(model_path) 
                          if d.startswith('checkpoint-') and 
                          os.path.isdir(os.path.join(model_path, d))]
        
        for ckpt_dir in checkpoint_dirs:
            try:
                step = int(ckpt_dir.split('-')[1])
            except (IndexError, ValueError):
                continue
            
            probing_csv = os.path.join(model_path, ckpt_dir, 'slurm_logs', 'probing_mmlu.csv')
            
            if os.path.exists(probing_csv):
                df = parse_probing_csv(probing_csv)
                if df is not None and not df.empty:
                    all_data[model_key][step] = {
                        'accuracy_per_layer': df['accuracy'].values,
                        'cv_mean_per_layer': df['cv_mean'].values,
                        'layers': df['layer'].values,
                        'training_type': training_type,
                        'dataset': dataset
                    }
    
    return all_data

def create_probing_accuracy_plots(all_data):
    """Create layerwise accuracy plots for each model."""
    output_dir = "/home/mmahaut/projects/paramem/plots_8b_probing"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_key, checkpoints in all_data.items():
        if not checkpoints:
            continue
        
        first_data = next(iter(checkpoints.values()))
        training_type = first_data['training_type']
        dataset = first_data['dataset']
        
        steps = sorted(checkpoints.keys())
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
        
        for idx, step in enumerate(steps):
            data = checkpoints[step]
            layers = data['layers']
            accuracy = data['accuracy_per_layer']
            
            ax.plot(layers, accuracy, 'o-', label=f'Step {step}', 
                   color=colors[idx], linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_title(f'Layerwise Probing Accuracy - {training_type} on {dataset.upper()}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Probing Accuracy', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        filename = f'{output_dir}/probing_accuracy_{model_key}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {filename}")

def create_probing_best_layer_plot(all_data):
    """Create plot showing best layer accuracy over training."""
    output_dir = "/home/mmahaut/projects/paramem/plots_8b_probing"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = sns.color_palette("husl", len(all_data))
    
    for idx, (model_key, checkpoints) in enumerate(all_data.items()):
        if not checkpoints:
            continue
        
        first_data = next(iter(checkpoints.values()))
        training_type = first_data['training_type']
        dataset = first_data['dataset']
        
        steps = sorted(checkpoints.keys())
        best_cv_accuracies = []
        
        for step in steps:
            data = checkpoints[step]
            best_cv = max(data['cv_mean_per_layer'])
            best_cv_accuracies.append(best_cv)
        
        label = f"{training_type} - {dataset.upper()}"
        ax.plot(steps, best_cv_accuracies, marker=markers[idx % len(markers)], 
               label=label, linewidth=2.5, markersize=8, color=colors[idx])
    
    ax.set_title('Best Layer CV Accuracy Evolution (Generalization)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Best Layer CV Mean Accuracy', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    filename = f'{output_dir}/probing_best_cv_accuracy_evolution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created {filename}")

def create_probing_heatmap(all_data):
    """Create heatmaps showing accuracy across layers and steps."""
    output_dir = "/home/mmahaut/projects/paramem/plots_8b_probing"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_key, checkpoints in all_data.items():
        if not checkpoints:
            continue
        
        first_data = next(iter(checkpoints.values()))
        training_type = first_data['training_type']
        dataset = first_data['dataset']
        
        steps = sorted(checkpoints.keys())
        num_layers = len(checkpoints[steps[0]]['layers'])
        
        # Create matrix: rows = steps, cols = layers
        accuracy_matrix = np.zeros((len(steps), num_layers))
        
        for step_idx, step in enumerate(steps):
            data = checkpoints[step]
            accuracy_matrix[step_idx, :] = data['accuracy_per_layer']
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        im = ax.imshow(accuracy_matrix, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest', vmin=0, vmax=1)
        
        ax.set_title(f'Probing Accuracy Heatmap - {training_type} on {dataset.upper()}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Training Step', fontsize=14)
        
        # Set y-ticks to actual step numbers
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(steps)
        
        # Set x-ticks for layers
        if num_layers > 20:
            ax.set_xticks(range(0, num_layers, 5))
        else:
            ax.set_xticks(range(num_layers))
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probing Accuracy', fontsize=12)
        
        plt.tight_layout()
        filename = f'{output_dir}/probing_heatmap_{model_key}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {filename}")

def create_cv_mean_plots(all_data):
    """Create plots showing CV mean (probing difficulty) across layers."""
    output_dir = "/home/mmahaut/projects/paramem/plots_8b_probing"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_key, checkpoints in all_data.items():
        if not checkpoints:
            continue
        
        first_data = next(iter(checkpoints.values()))
        training_type = first_data['training_type']
        dataset = first_data['dataset']
        
        steps = sorted(checkpoints.keys())
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))
        
        for idx, step in enumerate(steps):
            data = checkpoints[step]
            layers = data['layers']
            cv_mean = data['cv_mean_per_layer']
            
            ax.plot(layers, cv_mean, 'o-', label=f'Step {step}', 
                   color=colors[idx], linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_title(f'Probing Difficulty (CV Mean) - {training_type} on {dataset.upper()}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Cross-Validation Mean Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        filename = f'{output_dir}/probing_cv_mean_{model_key}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {filename}")

if __name__ == "__main__":
    print("🚀 Starting layerwise probing analysis...")
    
    # Collect probing data
    all_data = collect_probing_data()
    
    if not all_data:
        print("❌ No probing data found.")
        exit(1)
    
    print(f"\n📊 Summary of probing data:")
    for model_key, checkpoints in all_data.items():
        if checkpoints:
            steps = sorted(checkpoints.keys())
            print(f"   🔹 {model_key}: {len(steps)} checkpoints ({min(steps)}-{max(steps)})")
    
    print("\n📊 Creating probing analysis plots...")
    create_probing_accuracy_plots(all_data)
    create_probing_best_layer_plot(all_data)
    create_probing_heatmap(all_data)
    create_cv_mean_plots(all_data)
    
    print(f"\n🎉 Probing analysis complete!")
    print("📁 Plots saved to: /home/mmahaut/projects/paramem/plots_8b_probing/")
