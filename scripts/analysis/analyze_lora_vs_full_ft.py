#!/usr/bin/env python3
"""
Script to parse and visualize Intrinsic Dimension (ID) and Performance data
comparing LoRA vs Full Fine-tuning approaches across different datasets.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_checkpoint_from_path(path):
    """Extract checkpoint number from path."""
    match = re.search(r'checkpoint-(\d+)', path)
    return int(match.group(1)) if match else 0

def parse_model_config(path):
    """Extract model configuration from path."""
    parts = path.split('/')
    for part in parts:
        if 'Mistral-7B-v0.3-fsdp' in part:
            # Parse: Mistral-7B-v0.3-fsdp-{lora}-{dataset}-{lr}
            config_parts = part.split('-')
            fsdp_idx = config_parts.index('fsdp')
            lora = config_parts[fsdp_idx + 1]
            dataset = config_parts[fsdp_idx + 2]
            
            # Handle cases with and without lr specification
            lr_part = ""
            if len(config_parts) > fsdp_idx + 3:
                lr_part = f"_{config_parts[fsdp_idx + 3]}"
            
            training_type = 'LoRA' if lora == '1' else 'Full-FT'
            return {
                'training_type': training_type,
                'dataset': dataset,
                'lr': lr_part,
                'full_name': f"{training_type}_{dataset}{lr_part}"
            }
    return None

def parse_id_results(file_path):
    """Parse intrinsic dimension results from ID log file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find all benchmark results
        results = {}
        
        # Pattern to match: benchmark_name_intrinsic_dimension: followed by layer data
        pattern = r'(\w+)_intrinsic_dimension:\s*\n((?:\s+layer_\d+:.*\n)*)'
        matches = re.findall(pattern, content)
        
        for benchmark, layer_data in matches:
            layer_results = {}
            
            # Parse each layer line
            layer_lines = re.findall(r'layer_(\d+):\s*\[([\d\.\s]+)\]', layer_data)
            
            for layer_num, values in layer_lines:
                # Parse the values array
                values_clean = re.sub(r'\s+', ' ', values.strip())
                if values_clean:
                    try:
                        values_array = [float(x) for x in values_clean.split()]
                        # Take the last value (highest alpha) as the representative ID
                        layer_results[int(layer_num)] = values_array[-1] if values_array else 0.0
                    except ValueError:
                        continue
            
            if layer_results:
                results[benchmark] = layer_results
        
        return results
    except Exception as e:
        print(f"Error parsing ID file {file_path}: {e}")
        return {}

def parse_performance_results(file_path):
    """Parse performance results from performance log file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        results = {}
        
        # Pattern to match: benchmark_name: value
        pattern = r'(\w+):\s*([\d\.]+)'
        matches = re.findall(pattern, content)
        
        for benchmark, value in matches:
            try:
                results[benchmark] = float(value)
            except ValueError:
                continue
        
        return results
    except Exception as e:
        print(f"Error parsing performance file {file_path}: {e}")
        return {}

def collect_all_data():
    """Collect all ID and performance data from slurm_logs_copy."""
    base_path = "/home/mmahaut/projects/paramem/slurm_logs_copy/home/mmahaut/projects/paramem/models2"
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 Scanning for data in slurm_logs_copy/models2...")
    
    # Get all model directories
    model_dirs = []
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)) and 'Mistral-7B-v0.3-fsdp' in item:
            model_dirs.append(item)
    
    model_dirs.sort()
    print(f"📂 Found {len(model_dirs)} model directories: {model_dirs}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        
        # Get model config from directory name
        config = parse_model_config(model_path)
        if not config:
            continue
        
        model_key = config['full_name']
        
        # Find all checkpoints
        checkpoint_count = 0
        id_count = 0
        perf_count = 0
        
        for item in os.listdir(model_path):
            if item.startswith('checkpoint-') and item != 'checkpoint-seq':
                checkpoint_path = os.path.join(model_path, item)
                slurm_logs_path = os.path.join(checkpoint_path, 'slurm_logs')
                
                if os.path.exists(slurm_logs_path):
                    try:
                        checkpoint_num = int(item.split('-')[1])
                        checkpoint_count += 1
                    except ValueError:
                        continue  # Skip non-numeric checkpoint names
                    
                    # Store config info with each checkpoint
                    if checkpoint_num not in all_data[model_key]:
                        all_data[model_key][checkpoint_num] = {'config': config}
                    
                    # Process ID logs
                    id_out_file = os.path.join(slurm_logs_path, 'ID_matrixent.out')
                    if os.path.exists(id_out_file):
                        id_results = parse_id_results(id_out_file)
                        if id_results:
                            all_data[model_key][checkpoint_num]['id'] = id_results
                            id_count += 1
                    
                    # Process performance logs  
                    perf_out_file = os.path.join(slurm_logs_path, 'performance_evaluation.out')
                    if os.path.exists(perf_out_file):
                        perf_results = parse_performance_results(perf_out_file)
                        if perf_results:
                            all_data[model_key][checkpoint_num]['performance'] = perf_results
                            perf_count += 1
        
        print(f"📊 {model_key}: {checkpoint_count} checkpoints, {id_count} ID, {perf_count} perf")
    
    return all_data

def create_training_comparison_plots(all_data):
    """Create comparison plots between LoRA and Full Fine-tuning approaches."""
    
    # Separate data by training type and dataset
    lora_data = {}
    full_ft_data = {}
    
    for model_key, model_data in all_data.items():
        if not model_data:
            continue
        
        # Get config from first checkpoint
        first_checkpoint = next(iter(model_data.values()))
        config = first_checkpoint.get('config', {})
        
        if config.get('training_type') == 'LoRA':
            lora_data[model_key] = model_data
        elif config.get('training_type') == 'Full-FT':
            full_ft_data[model_key] = model_data
    
    print(f"\n📊 Found {len(lora_data)} LoRA models and {len(full_ft_data)} Full Fine-tuning models")
    
    # Create ID comparison plots
    create_id_comparison_plot(lora_data, full_ft_data, all_data)
    
    # Create performance comparison plots
    create_performance_comparison_plot(lora_data, full_ft_data, all_data)
    
    # Create layerwise comparison plots
    create_layerwise_comparison_plots(lora_data, full_ft_data)

def create_id_comparison_plot(lora_data, full_ft_data, all_data):
    """Create ID comparison plots between LoRA and Full FT."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Intrinsic Dimension: LoRA vs Full Fine-tuning Comparison', fontsize=18, fontweight='bold')
    
    benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 2, idx % 2]
        
        # Plot LoRA models
        for model_key, model_data in lora_data.items():
            config = next(iter(model_data.values())).get('config', {})
            dataset = config.get('dataset', 'unknown')
            
            checkpoints = sorted(model_data.keys())
            mean_ids = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = list(model_data[checkpoint]['id'][benchmark].values())
                    if layer_ids and any(id_val > 0 for id_val in layer_ids):
                        mean_id = np.mean([id_val for id_val in layer_ids if id_val > 0])
                        mean_ids.append(mean_id)
                        valid_checkpoints.append(checkpoint)
            
            if mean_ids and len(mean_ids) > 1:
                linestyle = '-' if dataset == 'pile' else '--'
                ax.plot(valid_checkpoints, mean_ids, 'o-', label=f'LoRA ({dataset})', 
                       linewidth=2, markersize=6, linestyle=linestyle, color='blue')
        
        # Plot Full FT models
        for model_key, model_data in full_ft_data.items():
            config = next(iter(model_data.values())).get('config', {})
            dataset = config.get('dataset', 'unknown')
            
            checkpoints = sorted(model_data.keys())
            mean_ids = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = list(model_data[checkpoint]['id'][benchmark].values())
                    if layer_ids and any(id_val > 0 for id_val in layer_ids):
                        mean_id = np.mean([id_val for id_val in layer_ids if id_val > 0])
                        mean_ids.append(mean_id)
                        valid_checkpoints.append(checkpoint)
            
            if mean_ids and len(mean_ids) > 1:
                linestyle = '-' if dataset == 'pile' else '--'
                ax.plot(valid_checkpoints, mean_ids, 's-', label=f'Full-FT ({dataset})', 
                       linewidth=2, markersize=6, linestyle=linestyle, color='red')
        
        ax.set_title(f'{benchmark.upper()} - Mean Intrinsic Dimension', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Mean ID', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/lora_vs_fullft_id_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created LoRA vs Full-FT ID comparison plot: lora_vs_fullft_id_comparison.png")

def create_performance_comparison_plot(lora_data, full_ft_data, all_data):
    """Create performance comparison plots between LoRA and Full FT."""
    fig, axes = plt.subplots(2, 3, figsize=(30, 18))
    fig.suptitle('Performance: LoRA vs Full Fine-tuning Comparison', fontsize=18, fontweight='bold')
    
    benchmarks = ['mmlu_accuracy', 'arc_accuracy', 'hellaswag_accuracy', 
                 'gsm8k_accuracy', 'truthfulqa_accuracy', 'openbookqa_accuracy']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 3, idx % 3]
        
        # Plot LoRA models
        for model_key, model_data in lora_data.items():
            config = next(iter(model_data.values())).get('config', {})
            dataset = config.get('dataset', 'unknown')
            
            checkpoints = sorted(model_data.keys())
            performance_values = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'performance' in model_data[checkpoint]:
                    perf_data = model_data[checkpoint]['performance']
                    if benchmark in perf_data and not np.isnan(perf_data[benchmark]) and perf_data[benchmark] > 0:
                        performance_values.append(perf_data[benchmark])
                        valid_checkpoints.append(checkpoint)
            
            if performance_values and len(performance_values) > 1:
                linestyle = '-' if dataset == 'pile' else '--'
                ax.plot(valid_checkpoints, performance_values, 'o-', label=f'LoRA ({dataset})', 
                       linewidth=2, markersize=6, linestyle=linestyle, color='blue')
        
        # Plot Full FT models
        for model_key, model_data in full_ft_data.items():
            config = next(iter(model_data.values())).get('config', {})
            dataset = config.get('dataset', 'unknown')
            
            checkpoints = sorted(model_data.keys())
            performance_values = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'performance' in model_data[checkpoint]:
                    perf_data = model_data[checkpoint]['performance']
                    if benchmark in perf_data and not np.isnan(perf_data[benchmark]) and perf_data[benchmark] > 0:
                        performance_values.append(perf_data[benchmark])
                        valid_checkpoints.append(checkpoint)
            
            if performance_values and len(performance_values) > 1:
                linestyle = '-' if dataset == 'pile' else '--'
                ax.plot(valid_checkpoints, performance_values, 's-', label=f'Full-FT ({dataset})', 
                       linewidth=2, markersize=6, linestyle=linestyle, color='red')
        
        benchmark_name = benchmark.replace('_accuracy', '').replace('_', ' ').upper()
        ax.set_title(f'{benchmark_name} Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/lora_vs_fullft_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created LoRA vs Full-FT performance comparison plot: lora_vs_fullft_performance_comparison.png")

def create_layerwise_comparison_plots(lora_data, full_ft_data):
    """Create layerwise ID comparison plots for final checkpoints."""
    
    # Get the latest checkpoint for each model
    lora_latest = {}
    full_ft_latest = {}
    
    for model_key, model_data in lora_data.items():
        if model_data:
            latest_checkpoint = max(model_data.keys())
            if 'id' in model_data[latest_checkpoint]:
                config = model_data[latest_checkpoint].get('config', {})
                dataset = config.get('dataset', 'unknown')
                lora_latest[f"{dataset}"] = model_data[latest_checkpoint]['id']
    
    for model_key, model_data in full_ft_data.items():
        if model_data:
            latest_checkpoint = max(model_data.keys())
            if 'id' in model_data[latest_checkpoint]:
                config = model_data[latest_checkpoint].get('config', {})
                dataset = config.get('dataset', 'unknown')
                full_ft_latest[f"{dataset}"] = model_data[latest_checkpoint]['id']
    
    # Create layerwise comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Layerwise ID Comparison: LoRA vs Full Fine-tuning (Latest Checkpoints)', fontsize=18, fontweight='bold')
    
    benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 2, idx % 2]
        
        # Determine max layer across all models
        max_layer = 0
        for dataset_data in [*lora_latest.values(), *full_ft_latest.values()]:
            if benchmark in dataset_data:
                max_layer = max(max_layer, max(dataset_data[benchmark].keys()) if dataset_data[benchmark] else 0)
        
        # Plot LoRA models
        for dataset, id_data in lora_latest.items():
            if benchmark in id_data:
                layers = list(range(max_layer + 1))
                id_values = [id_data[benchmark].get(layer, 0) for layer in layers]
                
                if any(val > 0 for val in id_values):
                    linestyle = '-' if dataset == 'pile' else '--'
                    ax.plot(layers, id_values, 'o-', label=f'LoRA ({dataset})', 
                           linewidth=2, markersize=4, linestyle=linestyle, color='blue')
        
        # Plot Full FT models
        for dataset, id_data in full_ft_latest.items():
            if benchmark in id_data:
                layers = list(range(max_layer + 1))
                id_values = [id_data[benchmark].get(layer, 0) for layer in layers]
                
                if any(val > 0 for val in id_values):
                    linestyle = '-' if dataset == 'pile' else '--'
                    ax.plot(layers, id_values, 's-', label=f'Full-FT ({dataset})', 
                           linewidth=2, markersize=4, linestyle=linestyle, color='red')
        
        ax.set_title(f'{benchmark.upper()} - Layerwise ID', fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Intrinsic Dimension', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-0.5, max_layer + 0.5)
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/lora_vs_fullft_layerwise_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created LoRA vs Full-FT layerwise comparison plot: lora_vs_fullft_layerwise_comparison.png")

def create_individual_plots_by_dataset(all_data):
    """Create individual plots separated by dataset and training type."""
    
    # Group by dataset
    datasets = defaultdict(lambda: {'LoRA': {}, 'Full-FT': {}})
    
    for model_key, model_data in all_data.items():
        if not model_data:
            continue
        
        # Get config from first checkpoint
        first_checkpoint = next(iter(model_data.values()))
        config = first_checkpoint.get('config', {})
        
        dataset = config.get('dataset', 'unknown')
        training_type = config.get('training_type', 'unknown')
        
        if training_type in ['LoRA', 'Full-FT']:
            datasets[dataset][training_type][model_key] = model_data
    
    # Create plots for each dataset
    for dataset_name, dataset_data in datasets.items():
        create_dataset_specific_plots(dataset_name, dataset_data)

def create_dataset_specific_plots(dataset_name, dataset_data):
    """Create plots specific to a dataset comparing LoRA vs Full-FT."""
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle(f'ID Evolution on {dataset_name.upper()} Dataset: LoRA vs Full Fine-tuning', fontsize=18, fontweight='bold')
    
    benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 2, idx % 2]
        
        # Plot LoRA models for this dataset
        for model_key, model_data in dataset_data['LoRA'].items():
            checkpoints = sorted(model_data.keys())
            mean_ids = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = list(model_data[checkpoint]['id'][benchmark].values())
                    if layer_ids and any(id_val > 0 for id_val in layer_ids):
                        mean_id = np.mean([id_val for id_val in layer_ids if id_val > 0])
                        mean_ids.append(mean_id)
                        valid_checkpoints.append(checkpoint)
            
            if mean_ids and len(mean_ids) > 1:
                ax.plot(valid_checkpoints, mean_ids, 'o-', label='LoRA', 
                       linewidth=3, markersize=6, color='blue')
        
        # Plot Full FT models for this dataset
        for model_key, model_data in dataset_data['Full-FT'].items():
            checkpoints = sorted(model_data.keys())
            mean_ids = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = list(model_data[checkpoint]['id'][benchmark].values())
                    if layer_ids and any(id_val > 0 for id_val in layer_ids):
                        mean_id = np.mean([id_val for id_val in layer_ids if id_val > 0])
                        mean_ids.append(mean_id)
                        valid_checkpoints.append(checkpoint)
            
            if mean_ids and len(mean_ids) > 1:
                ax.plot(valid_checkpoints, mean_ids, 's-', label='Full Fine-tuning', 
                       linewidth=3, markersize=6, color='red')
        
        ax.set_title(f'{benchmark.upper()} - Mean Intrinsic Dimension', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Mean ID', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'/home/mmahaut/projects/paramem/{dataset_name}_lora_vs_fullft_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created {dataset_name} dataset comparison plot: {dataset_name}_lora_vs_fullft_comparison.png")

if __name__ == "__main__":
    print("🚀 Starting LoRA vs Full Fine-tuning comparison analysis...")
    
    # Collect all data
    all_data = collect_all_data()
    
    if not all_data:
        print("❌ No data found. Check the log files and paths.")
        exit(1)
    
    print(f"\n📊 Found data for {len(all_data)} model configurations:")
    for model_key, model_data in all_data.items():
        if model_data:
            checkpoints = sorted(model_data.keys())
            id_count = sum(1 for cp in checkpoints if 'id' in model_data[cp])
            perf_count = sum(1 for cp in checkpoints if 'performance' in model_data[cp])
            config = next(iter(model_data.values())).get('config', {})
            training_type = config.get('training_type', 'unknown')
            dataset = config.get('dataset', 'unknown')
            print(f"   🔹 {training_type} on {dataset}: {len(checkpoints)} checkpoints ({min(checkpoints)}-{max(checkpoints)}), {id_count} ID, {perf_count} perf")
    
    print("\n📊 Creating comparison plots...")
    create_training_comparison_plots(all_data)
    
    print("📊 Creating dataset-specific plots...")
    create_individual_plots_by_dataset(all_data)
    
    print(f"\n🎉 Analysis complete! Generated plots saved in /home/mmahaut/projects/paramem/")
    print("📁 Files created:")
    print("   - lora_vs_fullft_id_comparison.png (ID comparison across all datasets)")
    print("   - lora_vs_fullft_performance_comparison.png (Performance comparison across all datasets)")
    print("   - lora_vs_fullft_layerwise_comparison.png (Layerwise ID comparison at final checkpoints)")
    print("   - pile_lora_vs_fullft_comparison.png (Pile dataset specific comparison)")
    print("   - wikiplus_lora_vs_fullft_comparison.png (Wikiplus dataset specific comparison)")
