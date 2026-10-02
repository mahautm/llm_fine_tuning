#!/usr/bin/env python3
"""
Script to parse and visualize Intrinsic Dimension (ID) and Performance data
from SLURM logs across all models and checkpoints.
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
            # Parse: Mistral-7B-v0.3-fsdp-{lora}-{dataset}-lr1e-4
            config_parts = part.split('-')
            lora_idx = config_parts.index('fsdp') + 1
            dataset_idx = lora_idx + 1
            
            lora = 'LoRA' if config_parts[lora_idx] == '1' else 'Full'
            dataset = config_parts[dataset_idx]
            
            return f"{lora}_{dataset}"
    return "unknown"

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
    """Collect all ID and performance data from log files."""
    base_path = "/home/mmahaut/projects/paramem/models_old"
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 Scanning for data in all model directories...")
    
    # Explicitly check each model directory
    model_dirs = [
        "Mistral-7B-v0.3-fsdp-0-pile-lr1e-4",
        "Mistral-7B-v0.3-fsdp-0-wikiplus-lr1e-4", 
        "Mistral-7B-v0.3-fsdp-1-pile-lr1e-4",
        "Mistral-7B-v0.3-fsdp-1-wikiplus-lr1e-4"
    ]
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        if not os.path.exists(model_path):
            print(f"⚠️  Model directory not found: {model_dir}")
            continue
            
        # Get model config from directory name
        config_parts = model_dir.split('-')
        lora_idx = config_parts.index('fsdp') + 1
        dataset_idx = lora_idx + 1
        
        lora = 'LoRA' if config_parts[lora_idx] == '1' else 'Full'
        dataset = config_parts[dataset_idx]
        model_config = f"{lora}_{dataset}"
        
        # Find all checkpoints
        checkpoint_count = 0
        id_count = 0
        perf_count = 0
        
        for item in os.listdir(model_path):
            if item.startswith('checkpoint-') and item != 'checkpoint-seq':
                checkpoint_path = os.path.join(model_path, item)
                slurm_logs_path = os.path.join(checkpoint_path, 'slurm_logs')
                
                if os.path.exists(slurm_logs_path):
                    checkpoint_num = int(item.split('-')[1])
                    checkpoint_count += 1
                    
                    # Process ID logs
                    id_out_file = os.path.join(slurm_logs_path, 'ID_matrixent.out')
                    if os.path.exists(id_out_file):
                        id_results = parse_id_results(id_out_file)
                        if id_results:
                            all_data[model_config][checkpoint_num]['id'] = id_results
                            id_count += 1
                    
                    # Process performance logs  
                    perf_out_file = os.path.join(slurm_logs_path, 'performance_evaluation.out')
                    if os.path.exists(perf_out_file):
                        perf_results = parse_performance_results(perf_out_file)
                        if perf_results:
                            all_data[model_config][checkpoint_num]['performance'] = perf_results
                            perf_count += 1
        
        print(f"📂 {model_config}: {checkpoint_count} checkpoints, {id_count} ID, {perf_count} perf")
    
    return all_data

def create_id_summary_plot(all_data):
    """Create summary plot of ID across all models and checkpoints."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Intrinsic Dimension Evolution Across Training - All Models', fontsize=16, fontweight='bold')
    
    model_configs = list(all_data.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_configs)))
    
    # Common benchmarks to plot
    benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 2, idx % 2]
        
        for model_idx, model_config in enumerate(model_configs):
            checkpoints = sorted(all_data[model_config].keys())
            mean_ids = []
            
            for checkpoint in checkpoints:
                if 'id' in all_data[model_config][checkpoint] and benchmark in all_data[model_config][checkpoint]['id']:
                    layer_ids = list(all_data[model_config][checkpoint]['id'][benchmark].values())
                    if layer_ids:
                        mean_id = np.mean([id_val for id_val in layer_ids if id_val > 0])
                        mean_ids.append(mean_id)
                    else:
                        mean_ids.append(np.nan)
                else:
                    mean_ids.append(np.nan)
            
            if mean_ids and not all(np.isnan(mean_ids)):
                ax.plot(checkpoints, mean_ids, 'o-', label=model_config, 
                       color=colors[model_idx], linewidth=2, markersize=8)
        
        ax.set_title(f'{benchmark.upper()} - Mean Intrinsic Dimension', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Mean ID', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/id_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created ID summary plot: id_summary_plot.png")

def create_performance_summary_plot(all_data):
    """Create summary plot of performance across all models and checkpoints."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Performance Evolution Across Training - All Models', fontsize=16, fontweight='bold')
    
    model_configs = list(all_data.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_configs)))
    
    # Common benchmarks to plot
    benchmarks = ['mmlu_accuracy', 'arc_accuracy', 'hellaswag_accuracy', 
                 'gsm8k_accuracy', 'truthfulqa_accuracy', 'openbookqa_accuracy']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 3, idx % 3]
        
        for model_idx, model_config in enumerate(model_configs):
            checkpoints = sorted(all_data[model_config].keys())
            performance_values = []
            
            for checkpoint in checkpoints:
                if 'performance' in all_data[model_config][checkpoint]:
                    perf_data = all_data[model_config][checkpoint]['performance']
                    performance_values.append(perf_data.get(benchmark, np.nan))
                else:
                    performance_values.append(np.nan)
            
            if performance_values and not all(np.isnan(performance_values)):
                ax.plot(checkpoints, performance_values, 'o-', label=model_config, 
                       color=colors[model_idx], linewidth=2, markersize=8)
        
        benchmark_name = benchmark.replace('_accuracy', '').replace('_', ' ').upper()
        ax.set_title(f'{benchmark_name} Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/performance_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created performance summary plot: performance_summary_plot.png")

def create_individual_plots(all_data):
    """Create individual plots for each model configuration."""
    
    for model_config in all_data.keys():
        model_data = all_data[model_config]
        checkpoints = sorted(model_data.keys())
        
        if len(checkpoints) < 1:
            continue
        
        # Create layerwise ID plot for this model (checkpoints in legend, layers on x-axis)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Layerwise Intrinsic Dimension - {model_config}', fontsize=16, fontweight='bold')
        
        benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k']
        
        for idx, benchmark in enumerate(benchmarks):
            ax = axes[idx // 2, idx % 2]
            
            # Get checkpoints that have ID data for this benchmark
            valid_checkpoints = []
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = model_data[checkpoint]['id'][benchmark]
                    if layer_ids and any(val > 0 for val in layer_ids.values()):
                        valid_checkpoints.append(checkpoint)
            
            if not valid_checkpoints:
                ax.text(0.5, 0.5, f'No {benchmark.upper()} data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{benchmark.upper()} - No Data', fontsize=14, fontweight='bold')
                continue
            
            # Determine max layer across all checkpoints
            max_layer = 0
            for checkpoint in valid_checkpoints:
                layer_ids = model_data[checkpoint]['id'][benchmark]
                max_layer = max(max_layer, max(layer_ids.keys()) if layer_ids else 0)
            
            # Plot each checkpoint as a line across layers
            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_checkpoints)))
            
            for cp_idx, checkpoint in enumerate(valid_checkpoints):
                layer_ids = model_data[checkpoint]['id'][benchmark]
                
                # Prepare data for plotting
                layers = list(range(max_layer + 1))
                id_values = [layer_ids.get(layer, 0) for layer in layers]
                
                # Only plot if we have meaningful data
                if any(val > 0 for val in id_values):
                    ax.plot(layers, id_values, 'o-', 
                           label=f'Checkpoint {checkpoint}', 
                           color=colors[cp_idx], 
                           linewidth=2, markersize=6)
            
            ax.set_title(f'{benchmark.upper()} - Layerwise ID', fontsize=14, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Intrinsic Dimension', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlim(-0.5, max_layer + 0.5)
        
        plt.tight_layout()
        plt.savefig(f'/home/mmahaut/projects/paramem/{model_config}_layerwise_id.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created layerwise ID plot: {model_config}_layerwise_id.png")
        
        # Create performance plot for this model
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(f'Performance Evolution - {model_config}', fontsize=16, fontweight='bold')
        
        perf_benchmarks = ['mmlu_accuracy', 'arc_accuracy', 'hellaswag_accuracy', 
                          'gsm8k_accuracy', 'truthfulqa_accuracy', 'openbookqa_accuracy']
        
        for idx, benchmark in enumerate(perf_benchmarks):
            ax = axes[idx // 3, idx % 3]
            
            performance_values = []
            valid_checkpoints = []
            
            for checkpoint in checkpoints:
                if 'performance' in model_data[checkpoint]:
                    perf_data = model_data[checkpoint]['performance']
                    if benchmark in perf_data and not np.isnan(perf_data[benchmark]):
                        performance_values.append(perf_data[benchmark])
                        valid_checkpoints.append(checkpoint)
            
            if len(performance_values) > 1:
                ax.plot(valid_checkpoints, performance_values, 'o-', 
                       linewidth=3, markersize=10, color='darkblue')
            
            benchmark_name = benchmark.replace('_accuracy', '').replace('_', ' ').upper()
            ax.set_title(f'{benchmark_name} Accuracy', fontsize=14, fontweight='bold')
            ax.set_xlabel('Checkpoint', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'/home/mmahaut/projects/paramem/{model_config}_performance_evolution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created performance evolution plot: {model_config}_performance_evolution.png")

def create_data_summary():
    """Create a summary of available data."""
    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    
    all_data = collect_all_data()
    
    total_models = len(all_data)
    total_checkpoints = sum(len(model_data) for model_data in all_data.values())
    
    print(f"📊 Found {total_models} model configurations")
    print(f"📊 Found {total_checkpoints} total checkpoints")
    print()
    
    for model_config, model_data in all_data.items():
        checkpoints = sorted(model_data.keys())
        id_count = sum(1 for cp in checkpoints if 'id' in model_data[cp])
        perf_count = sum(1 for cp in checkpoints if 'performance' in model_data[cp])
        
        print(f"🔹 {model_config}:")
        print(f"   Checkpoints: {len(checkpoints)} ({min(checkpoints)}-{max(checkpoints)})")
        print(f"   ID data: {id_count} checkpoints")
        print(f"   Performance data: {perf_count} checkpoints")
        print()
    
    return all_data

if __name__ == "__main__":
    print("🚀 Starting analysis of model checkpoints...")
    
    # Collect all data and create summary
    all_data = create_data_summary()
    
    if not all_data:
        print("❌ No data found. Check the log files and paths.")
        exit(1)
    
    print("📊 Creating summary plots...")
    create_id_summary_plot(all_data)
    create_performance_summary_plot(all_data)
    
    print("📊 Creating individual model plots...")
    create_individual_plots(all_data)
    
    print(f"\n🎉 Analysis complete! Generated plots saved in /home/mmahaut/projects/paramem/")
    print("📁 Files created:")
    print("   - id_summary_plot.png (Summary of all models' ID)")
    print("   - performance_summary_plot.png (Summary of all models' performance)")
    for model_config in all_data.keys():
        print(f"   - {model_config}_layerwise_id.png (Layerwise ID with checkpoints in legend)")
        print(f"   - {model_config}_performance_evolution.png (Performance over checkpoints)")