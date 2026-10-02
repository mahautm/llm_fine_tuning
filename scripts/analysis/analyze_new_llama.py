
#!/usr/bin/env python3
"""
Fresh analysis script for NEW Llama-3.2-1B models in models2 directory.
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

def collect_llama_data():
    """Collect data from NEW Llama-3.2-1B models."""
    base_path = "/home/mmahaut/projects/paramem/models2"
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 Analyzing NEW Llama-3.2-1B models in models2...")
    
    # Get Llama model directories only
    model_dirs = []
    for item in os.listdir(base_path):
        model_path = os.path.join(base_path, item)
        if os.path.isdir(model_path) and 'Llama-3.2-1B' in item:
            model_dirs.append(item)
    
    model_dirs.sort()
    print(f"📂 Found {len(model_dirs)} Llama model directories: {model_dirs}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        
        # Parse configuration from directory name
        if 'fsdp-1-' in model_dir:
            training_type = 'LoRA'
        elif 'fsdp-0-' in model_dir:
            training_type = 'Full-FT'
        else:
            continue
        
        if '-pile-' in model_dir:
            dataset = 'pile'
        elif '-wikiplus-' in model_dir:
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

def create_llama_id_plots(all_data):
    """Create ID comparison plots for Llama models."""
    
    # Separate LoRA and Full-FT
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
    
    # Create ID evolution plot - expanded to include all datasets
    fig, axes = plt.subplots(3, 4, figsize=(32, 24))
    fig.suptitle('Llama-3.2-1B: LoRA vs Full Fine-tuning ID Evolution (All Datasets)', fontsize=20, fontweight='bold')
    
    # Include standard benchmarks AND training/test datasets
    benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k', 
                  'train_train', 'test_train', 'wikidata_Mis7_train', 'pile_19_short_train',
                  'lambada', 'winogrande', 'openbookqa', 'truthful_qa']
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx // 4, idx % 4]
        
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
            
            if mean_ids and len(mean_ids) >= 1:
                linestyle = '-' if dataset == 'pile' else '--'
                marker = 'o'
                ax.plot(valid_checkpoints, mean_ids, marker=marker, linestyle=linestyle, 
                       label=f'LoRA ({dataset})', linewidth=3, markersize=8, color='blue')
        
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
            
            if mean_ids and len(mean_ids) >= 1:
                linestyle = '-' if dataset == 'pile' else '--'
                marker = 's'
                ax.plot(valid_checkpoints, mean_ids, marker=marker, linestyle=linestyle,
                       label=f'Full-FT ({dataset})', linewidth=3, markersize=8, color='red')
        
        ax.set_title(f'{benchmark.upper()} - Mean Intrinsic Dimension', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Mean ID', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/llama_lora_vs_fullft_id_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created Llama ID evolution plot: llama_lora_vs_fullft_id_evolution.png")

def create_llama_performance_plots(all_data):
    """Create performance comparison plots for Llama models."""
    
    # Separate LoRA and Full-FT
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
    
    # Create performance evolution plot
    fig, axes = plt.subplots(2, 3, figsize=(30, 18))
    fig.suptitle('Llama-3.2-1B: Performance Evolution Comparison', fontsize=18, fontweight='bold')
    
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
            
            if performance_values and len(performance_values) >= 1:
                linestyle = '-' if dataset == 'pile' else '--'
                ax.plot(valid_checkpoints, performance_values, 'o-', label=f'LoRA ({dataset})', 
                       linewidth=3, markersize=8, linestyle=linestyle, color='blue')
        
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
            
            if performance_values and len(performance_values) >= 1:
                linestyle = '-' if dataset == 'pile' else '--'
                ax.plot(valid_checkpoints, performance_values, 's-', label=f'Full-FT ({dataset})', 
                       linewidth=3, markersize=8, linestyle=linestyle, color='red')
        
        benchmark_name = benchmark.replace('_accuracy', '').replace('_', ' ').upper()
        ax.set_title(f'{benchmark_name} Accuracy', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/mmahaut/projects/paramem/llama_lora_vs_fullft_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created Llama performance plot: llama_lora_vs_fullft_performance.png")

def create_llama_layerwise_plots(all_data):
    """Create layerwise ID plots for each model and benchmark."""
    
    for model_key, model_data in all_data.items():
        if not model_data:
            continue
        
        config = next(iter(model_data.values())).get('config', {})
        training_type = config.get('training_type', 'unknown')
        dataset = config.get('dataset', 'unknown')
        
        checkpoints = sorted(model_data.keys())
        
        # Include standard benchmarks AND training/test datasets (including pile when available)
        benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k', 
                     'train_train', 'test_train', 'wikidata_Mis7_train',
                     'pile_19_short_train',
                     'lambada', 'winogrande', 'openbookqa', 'truthful_qa']
        
        for benchmark in benchmarks:
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
            for cp_idx, checkpoint in enumerate(checkpoints):
                if 'id' in model_data[checkpoint] and benchmark in model_data[checkpoint]['id']:
                    layer_ids = model_data[checkpoint]['id'][benchmark]
                    layers = list(range(max_layer + 1))
                    id_values = [layer_ids.get(layer, 0) for layer in layers]
                    
                    if any(val > 0 for val in id_values):
                        ax.plot(layers, id_values, 'o-', label=f'Step {checkpoint}', 
                               color=colors[cp_idx], linewidth=2, markersize=4)
            
            ax.set_title(f'Llama-3.2-1B {training_type} ({dataset}) - {benchmark.upper()} Layerwise ID', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=14)
            ax.set_ylabel('Intrinsic Dimension', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlim(-0.5, max_layer + 0.5)
            
            plt.tight_layout()
            filename = f'/home/mmahaut/projects/paramem/llama_{training_type.lower()}_{dataset}_{benchmark}_layerwise.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Created {filename}")

if __name__ == "__main__":
    print("🚀 Starting analysis of NEW Llama-3.2-1B models...")
    
    # Collect Llama data
    all_data = collect_llama_data()
    
    if not all_data:
        print("❌ No Llama data found.")
        exit(1)
    
    print(f"\n📊 Summary of Llama-3.2-1B models:")
    for model_key, model_data in all_data.items():
        if model_data:
            checkpoints = sorted(model_data.keys())
            id_count = sum(1 for cp in checkpoints if 'id' in model_data[cp])
            perf_count = sum(1 for cp in checkpoints if 'performance' in model_data[cp])
            config = next(iter(model_data.values())).get('config', {})
            print(f"   🔹 {model_key}: {len(checkpoints)} steps ({min(checkpoints)}-{max(checkpoints)}), {id_count} ID, {perf_count} perf")
    
    print("\n📊 Creating Llama analysis plots...")
    create_llama_id_plots(all_data)
    create_llama_performance_plots(all_data)
    create_llama_layerwise_plots(all_data)
    
    print(f"\n🎉 Llama-3.2-1B analysis complete!")
    print("📁 New plots generated:")
    print("   - llama_lora_vs_fullft_id_evolution.png")
    print("   - llama_lora_vs_fullft_performance.png")
    print("   - llama_[training_type]_[dataset]_[benchmark]_layerwise.png")