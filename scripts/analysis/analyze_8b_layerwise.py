#!/usr/bin/env python3
"""
Analysis script for Llama-3.1-8B models - creates layerwise ID plots for all checkpoints.
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
                        # Take the last value (highest alpha) as representative ID
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

def collect_8b_data():
    """Collect data from Llama-3.1-8B models."""
    base_path = "/home/mmahaut/projects/paramem/models2"
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 Analyzing Llama-3.1-8B models in models2...")
    
    # Get 8B model directories
    model_dirs = []
    for item in os.listdir(base_path):
        model_path = os.path.join(base_path, item)
        if os.path.isdir(model_path) and 'Llama-3.1-8B' in item:
            model_dirs.append(item)
    
    model_dirs.sort()
    print(f"📂 Found {len(model_dirs)} 8B model directories: {model_dirs}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        
        # Parse configuration from directory name
        if 'fsdp-1' in model_dir:
            training_type = 'LoRA'
        elif 'fsdp-0' in model_dir:
            training_type = 'Full-FT'
        else:
            continue
        
        if 'pile' in model_dir.lower():
            dataset = 'pile'
        elif 'wikiplus' in model_dir.lower():
            dataset = 'wikiplus'
        else:
            continue
        
        model_key = f"{training_type}_{dataset}"
        
        # Count data
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
                    
                    # Store config
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
        
        print(f"📊 {model_key}: {checkpoint_count} checkpoints, {id_count} ID, {perf_count} perf")
    
    return all_data

def create_8b_layerwise_plots(all_data):
    """Create layerwise ID plots for each 8B model and benchmark."""
    
    output_dir = "/home/mmahaut/projects/paramem/plots_8b_layerwise"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_key, model_data in all_data.items():
        if not model_data:
            continue
        
        config = next(iter(model_data.values())).get('config', {})
        training_type = config.get('training_type', 'unknown')
        dataset = config.get('dataset', 'unknown')
        
        checkpoints = sorted(model_data.keys())
        
        # Include all available benchmarks
        benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k', 
                     'train_train', 'test_train', 'test_test',
                     'wikidata_Mis7_train', 'pile_19_short_train',
                     'lambada', 'winogrande', 'openbookqa', 'truthfulqa']
        
        for benchmark in benchmarks:
            # Check if this benchmark has any data
            has_data = False
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    has_data = True
                    break
            
            if not has_data:
                continue
            
            fig, ax = plt.subplots(figsize=(18, 10))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoints)))
            max_layer = 0
            
            # Determine max layer
            for checkpoint in checkpoints:
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = model_data[checkpoint]['id'][benchmark]
                    if layer_ids:
                        max_layer = max(max_layer, max(layer_ids.keys()))
            
            # Plot each checkpoint
            plotted_any = False
            for cp_idx, checkpoint in enumerate(checkpoints):
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = model_data[checkpoint]['id'][benchmark]
                    layers = list(range(max_layer + 1))
                    id_values = [layer_ids.get(layer, 0) for layer in layers]
                    
                    if any(val > 0 for val in id_values):
                        ax.plot(layers, id_values, 'o-', label=f'Step {checkpoint}', 
                               color=colors[cp_idx], linewidth=2, markersize=4)
                        plotted_any = True
            
            if not plotted_any:
                plt.close()
                continue
            
            benchmark_title = benchmark.replace('_', ' ').title()
            ax.set_title(f'Llama-3.1-8B {training_type} ({dataset}) - {benchmark_title} Layerwise ID', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=14)
            ax.set_ylabel('Intrinsic Dimension', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.set_xlim(-0.5, max_layer + 0.5)
            
            plt.tight_layout()
            filename = f'{output_dir}/llama_8b_{training_type.lower()}_{dataset}_{benchmark}_layerwise.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Created {filename}")

def create_8b_performance_plots(all_data):
    """Create performance evolution plots for 8B models."""
    
    output_dir = "/home/mmahaut/projects/paramem/plots_8b_layerwise"
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Create performance evolution plot with publication-style formatting
    # Set publication-style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    
    benchmarks = ['mmlu_accuracy', 'arc_accuracy', 'hellaswag_accuracy', 
                 'gsm8k_accuracy', 'truthfulqa_accuracy', 'openbookqa_accuracy',
                 'lambada_accuracy', 'wikidata_Mis7_train_nwp', 'wikidata_Mis7_test_nwp']
    
    # Color palette for publication quality
    colors = {'LoRA_pile': '#2E86AB', 'LoRA_wikiplus': '#A23B72',
              'Full-FT_pile': '#F18F01', 'Full-FT_wikiplus': '#C73E1D'}
    markers = {'LoRA': 'o', 'Full-FT': 's'}
    
    # First pass: check which benchmarks have all-zero values
    benchmarks_to_plot = []
    skipped_benchmarks = []
    
    for benchmark in benchmarks:
        has_nonzero = False
        for model_key, model_data in {**lora_data, **full_ft_data}.items():
            for checkpoint in model_data.keys():
                if 'performance' in model_data[checkpoint]:
                    perf_data = model_data[checkpoint]['performance']
                    if benchmark in perf_data and not np.isnan(perf_data[benchmark]) and perf_data[benchmark] > 0:
                        has_nonzero = True
                        break
            if has_nonzero:
                break
        
        if has_nonzero:
            benchmarks_to_plot.append(benchmark)
        else:
            skipped_benchmarks.append(benchmark)
    
    # Calculate grid size based on number of plots
    n_plots = len(benchmarks_to_plot)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Performance Evolution: LoRA vs Full Fine-Tuning', fontsize=14, fontweight='bold', y=0.995)
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, benchmark in enumerate(benchmarks_to_plot):
        ax = axes[idx]
        
        # Collect all data for this benchmark
        plot_data = []
        
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
                    if benchmark in perf_data and not np.isnan(perf_data[benchmark]):
                        performance_values.append(perf_data[benchmark])
                        valid_checkpoints.append(checkpoint)
            
            if performance_values and len(performance_values) >= 1:
                # Normalize: earliest checkpoint = 0, latest checkpoint = 1
                if len(valid_checkpoints) > 1:
                    min_cp = min(valid_checkpoints)
                    max_cp = max(valid_checkpoints)
                    valid_checkpoints_normalized = [(cp - min_cp) / (max_cp - min_cp) for cp in valid_checkpoints]
                else:
                    valid_checkpoints_normalized = [0.5]
                
                color_key = f'LoRA_{dataset}'
                ax.plot(valid_checkpoints_normalized, performance_values, 
                       marker=markers['LoRA'], label=f'LoRA ({dataset})', 
                       linewidth=2, markersize=6, color=colors.get(color_key, '#2E86AB'),
                       markeredgewidth=1.5, markeredgecolor='white', alpha=0.8)
        
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
                    if benchmark in perf_data and not np.isnan(perf_data[benchmark]):
                        performance_values.append(perf_data[benchmark])
                        valid_checkpoints.append(checkpoint)
            
            if performance_values and len(performance_values) >= 1:
                # Normalize: earliest checkpoint = 0, latest checkpoint = 1
                if len(valid_checkpoints) > 1:
                    min_cp = min(valid_checkpoints)
                    max_cp = max(valid_checkpoints)
                    valid_checkpoints_normalized = [(cp - min_cp) / (max_cp - min_cp) for cp in valid_checkpoints]
                else:
                    valid_checkpoints_normalized = [0.5]
                
                color_key = f'Full-FT_{dataset}'
                ax.plot(valid_checkpoints_normalized, performance_values, 
                       marker=markers['Full-FT'], label=f'Full-FT ({dataset})', 
                       linewidth=2, markersize=6, color=colors.get(color_key, '#F18F01'),
                       markeredgewidth=1.5, markeredgecolor='white', alpha=0.8)
        
        benchmark_name = benchmark.replace('_accuracy', '').replace('wikidata_Mis7_', '').replace('_nwp', '').replace('_', ' ').upper()
        ax.set_title(benchmark_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Normalized Training Progress', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='--')
        ax.legend(fontsize=8, frameon=True, fancybox=True, shadow=True, loc='best')
        ax.set_xlim(-0.05, 1.05)
        
        # Auto-scale y-axis for better visualization, color-code to indicate different scales
        # Use automatic scaling instead of fixed [0, 1]
        ax.autoscale(enable=True, axis='y', tight=False)
        y_min, y_max = ax.get_ylim()
        
        # Add some padding to y-axis
        y_range = y_max - y_min
        ax.set_ylim(max(0, y_min - 0.05 * y_range), min(1, y_max + 0.05 * y_range))
        
        # Color y-axis to indicate custom scaling (different from standard [0,1])
        if y_max < 0.95 or y_min > 0.05:  # If not using full [0,1] range
            ax.yaxis.label.set_color('#D32F2F')  # Red color for custom scale
            ax.tick_params(axis='y', colors='#D32F2F')
            # Add subtle background to indicate zoomed-in view
            ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.05, color='red', zorder=-1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(len(benchmarks_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Add text box for skipped benchmarks if any (after tight_layout to avoid overlap)
    if skipped_benchmarks:
        skip_text = "Skipped (all zeros):\n" + ", ".join([b.replace('_accuracy', '').replace('_', ' ').upper() for b in skipped_benchmarks])
        fig.text(0.98, 0.98, skip_text, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='gray', linewidth=0.8),
                transform=fig.transFigure)
    filename = f'{output_dir}/llama_8b_lora_vs_fullft_performance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created {filename}")

if __name__ == "__main__":
    print("🚀 Starting analysis of Llama-3.1-8B models...")
    
    # Collect 8B data
    all_data = collect_8b_data()
    
    if not all_data:
        print("❌ No 8B data found.")
        exit(1)
    
    print(f"\n📊 Summary of Llama-3.1-8B models:")
    for model_key, model_data in all_data.items():
        if model_data:
            checkpoints = sorted(model_data.keys())
            id_count = sum(1 for cp in checkpoints if 'id' in model_data[cp])
            perf_count = sum(1 for cp in checkpoints if 'performance' in model_data[cp])
            config = next(iter(model_data.values())).get('config', {})
            print(f"   🔹 {model_key}: {len(checkpoints)} steps ({min(checkpoints)}-{max(checkpoints)}), {id_count} ID, {perf_count} perf")
    
    print("\n📊 Creating 8B analysis plots...")
    create_8b_layerwise_plots(all_data)
    create_8b_performance_plots(all_data)
    
    print(f"\n🎉 Llama-3.1-8B analysis complete!")
    print("📁 Plots saved to: /home/mmahaut/projects/paramem/plots_8b_layerwise/")
