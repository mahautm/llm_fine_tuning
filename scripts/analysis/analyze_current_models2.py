#!/usr/bin/env python3
"""
Comprehensive analysis script for current models2 directory data.
Generates all plots for LoRA vs Full Fine-tuning comparison.
"""

import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_id_results(file_path):
    """Parse intrinsic dimension results from ID log file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        results = {}
        pattern = r'(\w+)_intrinsic_dimension:\s*\n((?:\s+layer_\d+:.*\n)*)'
        matches = re.findall(pattern, content)
        
        for benchmark, layer_data in matches:
            layer_results = {}
            layer_lines = re.findall(r'layer_(\d+):\s*\[([\d\.\s]+)\]', layer_data)
            
            for layer_num, values in layer_lines:
                values_clean = re.sub(r'\s+', ' ', values.strip())
                if values_clean:
                    try:
                        values_array = [float(x) for x in values_clean.split()]
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

def analyze_models2_direct():
    """Analyze data directly from /home/mmahaut/projects/paramem/models2."""
    base_path = "/home/mmahaut/projects/paramem/models2"
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 Scanning models2 directory directly...")
    
    if not os.path.exists(base_path):
        print(f"❌ Directory not found: {base_path}")
        return {}
    
    # Get all model directories
    model_dirs = []
    for item in os.listdir(base_path):
        model_path = os.path.join(base_path, item)
        if os.path.isdir(model_path):
            model_dirs.append(item)
    
    model_dirs.sort()
    print(f"📂 Found {len(model_dirs)} model directories: {model_dirs}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        
        # Parse model configuration from directory name
        if 'fsdp-1-' in model_dir:
            training_type = 'LoRA'
        elif 'fsdp-0-' in model_dir:
            training_type = 'Full-FT'
        else:
            continue
        
        # Extract dataset
        if '-pile-' in model_dir:
            dataset = 'pile'
        elif '-wikiplus-' in model_dir:
            dataset = 'wikiplus'
        else:
            continue
        
        model_key = f"{training_type}_{dataset}"
        
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
                        continue
                    
                    # Store config info with each checkpoint
                    if checkpoint_num not in all_data[model_key]:
                        all_data[model_key][checkpoint_num] = {
                            'config': {
                                'training_type': training_type,
                                'dataset': dataset,
                                'model_dir': model_dir
                            }
                        }
                    
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
        
        print(f"📊 {model_key} ({model_dir}): {checkpoint_count} checkpoints, {id_count} ID, {perf_count} perf")
    
    return all_data

def create_comprehensive_comparison_plots(all_data):
    """Create all comparison plots."""
    
    # Separate data by training type
    lora_data = {}
    full_ft_data = {}
    
    for model_key, model_data in all_data.items():
        if not model_data:
            continue
        
        first_checkpoint = next(iter(model_data.values()))
        config = first_checkpoint.get('config', {})
        
        if config.get('training_type') == 'LoRA':
            lora_data[model_key] = model_data
        elif config.get('training_type') == 'Full-FT':
            full_ft_data[model_key] = model_data
    
    print(f"\n📊 Found {len(lora_data)} LoRA models and {len(full_ft_data)} Full Fine-tuning models")
    
    # Create ID comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Intrinsic Dimension: LoRA vs Full Fine-tuning (Current Models2)', fontsize=18, fontweight='bold')
    
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
    plt.savefig('/home/mmahaut/projects/paramem/current_models2_id_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created current models2 ID comparison plot: current_models2_id_comparison.png")

def create_summary_table(all_data):
    """Create a summary table of all available data."""
    print("\n📋 SUMMARY OF AVAILABLE DATA:")
    print("=" * 80)
    
    for model_key, model_data in sorted(all_data.items()):
        if not model_data:
            continue
        
        first_checkpoint = next(iter(model_data.values()))
        config = first_checkpoint.get('config', {})
        
        checkpoints = sorted(model_data.keys())
        id_count = sum(1 for cp in checkpoints if 'id' in model_data[cp])
        perf_count = sum(1 for cp in checkpoints if 'performance' in model_data[cp])
        
        print(f"🔹 {model_key}")
        print(f"   Model Dir: {config.get('model_dir', 'unknown')}")
        print(f"   Checkpoints: {len(checkpoints)} total ({min(checkpoints)}-{max(checkpoints)})")
        print(f"   ID Data: {id_count} checkpoints")
        print(f"   Performance Data: {perf_count} checkpoints")
        print()

if __name__ == "__main__":
    print("🚀 Starting comprehensive analysis of current models2 directory...")
    
    # Analyze current models2 data
    all_data = analyze_models2_direct()
    
    if not all_data:
        print("❌ No data found in models2 directory.")
        exit(1)
    
    # Create summary
    create_summary_table(all_data)
    
    # Create plots
    print("📊 Creating comprehensive comparison plots...")
    create_comprehensive_comparison_plots(all_data)
    
    print(f"\n🎉 Analysis complete! Check /home/mmahaut/projects/paramem/ for:")
    print("   - current_models2_id_comparison.png")