#!/usr/bin/env python3
"""
Compare intrinsic dimension (models3) with memorization metrics (models4 v3).
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

def extract_models3_id(training_type="LoRA"):
    """Extract average layer-wise ID from models3 for given training type (LoRA or Full)."""
    suffix = "fsdp-1" if training_type.lower() == "lora" else "fsdp-0"
    id_files = sorted(glob.glob(f"models3/Llama-3.1-8B-Instruct-{suffix}-wikiplus-lr1e-4/checkpoint-*/slurm_logs/ID_matrixent.out"))
    id_data = {}

    for f in id_files:
        ckpt = int(os.path.basename(os.path.dirname(os.path.dirname(f))).replace("checkpoint-", ""))
        with open(f) as fh:
            content = fh.read()
            # Look for pile_19_short_train intrinsic_dimension (the training dataset)
            pattern = r"pile_19_short_train_intrinsic_dimension:\s*\n((?:\s+layer_\d+:.*\n)*)"
            matches = re.findall(pattern, content)

            if matches:
                layer_data = matches[0]
                layer_lines = re.findall(r"layer_(\d+):\s*\[([\d\.\s]+)\]", layer_data)
                all_ids = []

                for layer_num, values in layer_lines:
                    values_clean = re.sub(r"\s+", " ", values.strip())
                    if values_clean:
                        try:
                            values_array = [float(x) for x in values_clean.split()]
                            # Take the last value (highest alpha)
                            if values_array:
                                all_ids.append(values_array[-1])
                        except ValueError:
                            continue

                if all_ids:
                    avg_id = sum(all_ids) / len(all_ids)
                    id_data[ckpt] = avg_id

    return id_data

def extract_models4_memorization(training_type="LoRA"):
    """Extract memorization metrics from models4 v3 for given training type."""
    suffix = "fsdp-1" if training_type.lower() == "lora" else "fsdp-0"
    mem_files = sorted(glob.glob(f"models4/Llama-3.1-8B-Instruct-{suffix}-wikiplus-lr1e-4-v3/checkpoint-*/slurm_logs/memorization_metrics.json"))
    mem_data = {}

    for f in mem_files:
        ckpt = int(os.path.basename(os.path.dirname(os.path.dirname(f))).replace("checkpoint-", ""))
        with open(f) as fh:
            m = json.load(fh)
            mem_data[ckpt] = {
                'extractability_rate': m.get('extractability_rate', 0),
                'train_nll': m.get('train_nll', 0),
                'train_perplexity': m.get('train_perplexity', 0),
                'exposure_delta': m.get('exposure_delta', 0),
                'intrinsic_dimension': m.get('intrinsic_dimension', 0),
            }

    return mem_data

def gather_run(training_type):
    id_data = extract_models3_id(training_type)
    mem_data = extract_models4_memorization(training_type)
    common_steps = sorted(set(id_data.keys()) & set(mem_data.keys()))
    return id_data, mem_data, common_steps

def plot_comparison():
    """Create comparison plots of ID vs memorization metrics for LoRA and Full."""
    runs = {
        "LoRA": {
            "id_data": None,
            "mem_data": None,
            "steps": None,
            "id_color": "#2E86AB",
            "metric_color": "#D62728",
            "style": "-",
        },
        "Full": {
            "id_data": None,
            "mem_data": None,
            "steps": None,
            "id_color": "#9467BD",
            "metric_color": "#FF7F0E",
            "style": "--",
        },
    }

    for name in runs.keys():
        id_data, mem_data, steps = gather_run(name)
        runs[name]["id_data"] = id_data
        runs[name]["mem_data"] = mem_data
        runs[name]["steps"] = steps

    if not runs["LoRA"]["steps"] and not runs["Full"]["steps"]:
        print("❌ No common checkpoints found for either run!")
        return

    def plot_panel(ax, metric_key, title):
        ax2 = ax.twinx()
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Avg Layer ID', color='black')
        ax2.set_ylabel(metric_key.replace('_', ' ').title(), color='black')
        ax.grid(True, alpha=0.3)

        first_corr = None
        for name, cfg in runs.items():
            steps = cfg["steps"]
            if not steps:
                continue
            id_vals = [cfg["id_data"][s] for s in steps]
            metric_vals = [cfg["mem_data"][s][metric_key] for s in steps]

            ax.plot(steps, id_vals, cfg["style"], color=cfg["id_color"], linewidth=2, marker='o', markersize=5, label=f"{name} ID")
            ax2.plot(steps, metric_vals, cfg["style"], color=cfg["metric_color"], linewidth=2, marker='s', markersize=5, label=f"{name} {metric_key}")

            if first_corr is None:
                first_corr = np.corrcoef(id_vals, metric_vals)[0, 1]

        if first_corr is not None:
            ax.set_title(f"{title} (r={first_corr:.3f})")

        handles, labels = [], []
        for axis in [ax, ax2]:
            h, l = axis.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Intrinsic Dimension (models3) vs Memorization Metrics (models4 v3)\nLoRA (fsdp-1) vs Full (fsdp-0) - Wikiplus',
                 fontsize=14, fontweight='bold')

    plot_panel(axes[0, 0], 'extractability_rate', 'ID vs Extractability')
    plot_panel(axes[0, 1], 'train_nll', 'ID vs Train NLL')
    plot_panel(axes[1, 0], 'train_perplexity', 'ID vs Perplexity')
    plot_panel(axes[1, 1], 'exposure_delta', 'ID vs Exposure Delta')

    plt.tight_layout()

    output_path = "plots_memorization/id_memorization_correlation.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for name, cfg in runs.items():
        steps = cfg["steps"]
        if not steps:
            print(f"{name}: no overlapping checkpoints")
            continue
        ids = [cfg["id_data"][s] for s in steps]
        mem = cfg["mem_data"]
        print(f"{name}: steps {steps[0]} → {steps[-1]}")
        print(f"  ID: {ids[0]:.2f} → {ids[-1]:.2f} ({100*(ids[-1]/ids[0]-1):.1f}% change)")
        print(f"  Extractability: {mem[steps[0]]['extractability_rate']:.4f} → {mem[steps[-1]]['extractability_rate']:.4f}")
        print(f"  Train NLL: {mem[steps[0]]['train_nll']:.3f} → {mem[steps[-1]]['train_nll']:.3f}")
        print(f"  Perplexity: {mem[steps[0]]['train_perplexity']:.2f} → {mem[steps[-1]]['train_perplexity']:.2f}")
        print(f"  Exposure Δ: {mem[steps[0]]['exposure_delta']:.2f} → {mem[steps[-1]]['exposure_delta']:.2f}")

if __name__ == "__main__":
    plot_comparison()
