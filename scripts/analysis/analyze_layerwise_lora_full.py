#!/usr/bin/env python3
"""
Script to generate layerwise ID plots for each checkpoint, separated for LoRA and Full Fine-tuning models.
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

base_path = "/home/mmahaut/projects/paramem/slurm_logs_copy/home/mmahaut/projects/paramem/models2"

# Helper functions

def parse_model_config(path):
    parts = path.split('/')
    for part in parts:
        if 'Mistral-7B-v0.3-fsdp' in part:
            config_parts = part.split('-')
            fsdp_idx = config_parts.index('fsdp')
            lora = config_parts[fsdp_idx + 1]
            dataset = config_parts[fsdp_idx + 2]
            training_type = 'LoRA' if lora == '1' else 'Full-FT'
            return training_type, dataset, part
    return None, None, None

def parse_id_results(file_path):
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

def collect_layerwise_data(training_type):
    all_data = defaultdict(lambda: defaultdict(dict))
    for model_dir in os.listdir(base_path):
        model_path = os.path.join(base_path, model_dir)
        if not os.path.isdir(model_path):
            continue
        ttype, dataset, _ = parse_model_config(model_path)
        if ttype != training_type:
            continue
        for item in os.listdir(model_path):
            if item.startswith('checkpoint-') and item != 'checkpoint-seq':
                checkpoint_path = os.path.join(model_path, item)
                slurm_logs_path = os.path.join(checkpoint_path, 'slurm_logs')
                if os.path.exists(slurm_logs_path):
                    try:
                        checkpoint_num = int(item.split('-')[1])
                    except ValueError:
                        continue
                    id_out_file = os.path.join(slurm_logs_path, 'ID_matrixent.out')
                    if os.path.exists(id_out_file):
                        id_results = parse_id_results(id_out_file)
                        if id_results:
                            all_data[dataset][checkpoint_num] = id_results
    return all_data

def plot_layerwise_id(training_type, all_data):
    benchmarks = ['mmlu', 'arc', 'hellaswag', 'gsm8k']
    for dataset, checkpoints in all_data.items():
        for benchmark in benchmarks:
            fig, ax = plt.subplots(figsize=(18, 10))
            cp_sorted = sorted(checkpoints.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(cp_sorted)))
            max_layer = 0
            for cp_idx, checkpoint in enumerate(cp_sorted):
                id_data = checkpoints[checkpoint]
                if benchmark in id_data:
                    layer_ids = id_data[benchmark]
                    layers = list(layer_ids.keys())
                    max_layer = max(max_layer, max(layers))
            for cp_idx, checkpoint in enumerate(cp_sorted):
                id_data = checkpoints[checkpoint]
                if benchmark in id_data:
                    layer_ids = id_data[benchmark]
                    layers = list(range(max_layer + 1))
                    id_values = [layer_ids.get(layer, 0) for layer in layers]
                    if any(val > 0 for val in id_values):
                        ax.plot(layers, id_values, 'o-', label=f'CP {checkpoint}', color=colors[cp_idx], linewidth=2, markersize=4)
            ax.set_title(f'{training_type} - {dataset} - {benchmark.upper()} Layerwise ID', fontsize=16, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=14)
            ax.set_ylabel('Intrinsic Dimension', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlim(-0.5, max_layer + 0.5)
            plt.tight_layout()
            fname = f'/home/mmahaut/projects/paramem/{training_type.lower()}_{dataset}_{benchmark}_layerwise_id.png'
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Created {fname}")

if __name__ == "__main__":
    for ttype in ['LoRA', 'Full-FT']:
        print(f"\nCollecting layerwise ID data for {ttype} models...")
        data = collect_layerwise_data(ttype)
        plot_layerwise_id(ttype, data)
    print("\n🎉 Layerwise ID plots generated for LoRA and Full Fine-tuning models!")
