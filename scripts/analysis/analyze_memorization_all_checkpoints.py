#!/usr/bin/env python3
"""
Collect and analyze memorization metrics from all evaluated checkpoints.
Creates comprehensive plots showing memorization evolution across training.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import defaultdict

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def collect_memorization_results():
    """Collect all memorization metrics from checkpoint evaluations."""
    
    models_dir = Path("/home/mmahaut/projects/paramem/models3")
    
    all_results = []
    
    # Process each model directory
    for model_dir in models_dir.glob("Llama-3.1-8B*"):
        model_name = model_dir.name
        
        # Parse model configuration
        if 'fsdp-1' in model_name:
            training_type = 'LoRA'
        elif 'fsdp-0' in model_name:
            training_type = 'Full-FT'
        else:
            continue
        
        if 'pile' in model_name.lower():
            dataset = 'Pile'
        elif 'wikiplus' in model_name.lower():
            dataset = 'Wikiplus'
        elif 'mmlu' in model_name.lower():
            dataset = 'MMLU'
        else:
            continue
        
        # Find all checkpoints with memorization results
        for checkpoint_dir in sorted(model_dir.glob("checkpoint-*")):
            if not checkpoint_dir.is_dir():
                continue
            
            try:
                checkpoint_num = int(checkpoint_dir.name.split('-')[1])
            except (IndexError, ValueError):
                continue
            
            # Check for memorization metrics
            mem_json = checkpoint_dir / "slurm_logs" / "memorization_metrics.json"
            
            if mem_json.exists():
                try:
                    with open(mem_json) as f:
                        metrics = json.load(f)
                    
                    result = {
                        'model': model_name,
                        'training_type': training_type,
                        'dataset': dataset,
                        'checkpoint': checkpoint_num,
                        'extractability_rate': metrics.get('extractability_rate', 0.0),
                        'avg_tokens_recovered': metrics.get('avg_tokens_recovered', 0.0),
                        'ngram_correlation': metrics.get('ngram_correlation', 0.0),
                        'intrinsic_dimension': metrics.get('intrinsic_dimension', 0.0),
                        'num_sequences_tested': metrics.get('num_sequences_tested', 0)
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error loading {mem_json}: {e}")
    
    return pd.DataFrame(all_results)


def create_memorization_plots(df, output_dir="memorization_analysis_all"):
    """Create comprehensive plots for memorization analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if df.empty:
        print("No data to plot!")
        return
    
    print(f"\nCreating plots with {len(df)} data points...")
    
    # 1. Evolution plots by training type and dataset
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Memorization Metrics Evolution Across Training", fontsize=16, y=0.995)
    
    metrics_to_plot = [
        ('extractability_rate', 'Discoverable Memorization\n(Extractability Rate)', axes[0, 0]),
        ('avg_tokens_recovered', 'Token Recovery Rate', axes[0, 1]),
        ('ngram_correlation', 'Distributional Memorization\n(N-gram Correlation)', axes[1, 0]),
        ('intrinsic_dimension', 'Intrinsic Dimension\n(Embedding Complexity)', axes[1, 1])
    ]
    
    for metric_name, title, ax in metrics_to_plot:
        for (training_type, dataset), group in df.groupby(['training_type', 'dataset']):
            if len(group) > 0:
                group_sorted = group.sort_values('checkpoint')
                label = f"{training_type} - {dataset}"
                ax.plot(group_sorted['checkpoint'], group_sorted[metric_name], 
                       marker='o', label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Training Step (Checkpoint)", fontsize=11)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "memorization_evolution_all.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: memorization_evolution_all.png")
    plt.close()
    
    # 2. ID vs Memorization scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Hypothesis Test: Higher ID → Harder to Memorize", fontsize=16)
    
    scatter_metrics = [
        ('extractability_rate', 'Extractability Rate'),
        ('avg_tokens_recovered', 'Avg Tokens Recovered'),
        ('ngram_correlation', 'N-gram Correlation')
    ]
    
    for idx, (metric_name, ylabel) in enumerate(scatter_metrics):
        ax = axes[idx]
        
        for (training_type, dataset), group in df.groupby(['training_type', 'dataset']):
            if len(group) > 1:
                label = f"{training_type} - {dataset}"
                scatter = ax.scatter(
                    group['intrinsic_dimension'],
                    group[metric_name],
                    s=80,
                    alpha=0.7,
                    label=label
                )
        
        # Overall trend line
        if len(df) > 2:
            valid_data = df[['intrinsic_dimension', metric_name]].dropna()
            if len(valid_data) > 2:
                z = np.polyfit(valid_data['intrinsic_dimension'], valid_data[metric_name], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid_data['intrinsic_dimension'].min(), 
                                     valid_data['intrinsic_dimension'].max(), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label='Overall Trend')
                
                # Correlation
                corr = np.corrcoef(valid_data['intrinsic_dimension'], valid_data[metric_name])[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}',
                       transform=ax.transAxes, fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel("Intrinsic Dimension", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "id_vs_memorization_all.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: id_vs_memorization_all.png")
    plt.close()
    
    # 3. Comparison by training type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("LoRA vs Full Fine-Tuning: Memorization Comparison", fontsize=16, y=0.995)
    
    for idx, (metric_name, title) in enumerate([
        ('extractability_rate', 'Extractability Rate'),
        ('avg_tokens_recovered', 'Token Recovery'),
        ('ngram_correlation', 'N-gram Correlation'),
        ('intrinsic_dimension', 'Intrinsic Dimension')
    ]):
        ax = axes[idx // 2, idx % 2]
        
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            
            for training_type in ['LoRA', 'Full-FT']:
                type_data = dataset_data[dataset_data['training_type'] == training_type]
                if len(type_data) > 0:
                    type_sorted = type_data.sort_values('checkpoint')
                    linestyle = '-' if training_type == 'Full-FT' else '--'
                    label = f"{dataset} ({training_type})"
                    ax.plot(type_sorted['checkpoint'], type_sorted[metric_name],
                           marker='o', label=label, linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "lora_vs_fullft_memorization.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: lora_vs_fullft_memorization.png")
    plt.close()
    
    # 4. Summary statistics table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary by model
    summary_data = []
    for (training_type, dataset), group in df.groupby(['training_type', 'dataset']):
        if len(group) > 0:
            summary_data.append([
                f"{training_type} - {dataset}",
                len(group),
                f"{group['extractability_rate'].mean():.4f} ± {group['extractability_rate'].std():.4f}",
                f"{group['avg_tokens_recovered'].mean():.4f} ± {group['avg_tokens_recovered'].std():.4f}",
                f"{group['ngram_correlation'].mean():.4f} ± {group['ngram_correlation'].std():.4f}",
                f"{group['intrinsic_dimension'].mean():.2f} ± {group['intrinsic_dimension'].std():.2f}"
            ])
    
    headers = ['Model', 'N', 'Extractability', 'Token Recovery', 'N-gram Corr', 'ID']
    table = ax.table(cellText=summary_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title("Memorization Metrics Summary Statistics (Mean ± Std)", 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path / "memorization_summary_table.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: memorization_summary_table.png")
    plt.close()


def main():
    """Main execution."""
    print("="*60)
    print("Memorization Analysis: Collecting Results")
    print("="*60)
    
    df = collect_memorization_results()
    
    if df.empty:
        print("\n⚠ No memorization results found!")
        print("Make sure to run: ./scripts/entrypoints/jobs/run_memorization_on_checkpoints.sh")
        return
    
    print(f"\n✓ Collected {len(df)} checkpoint results")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Training types: {df['training_type'].unique().tolist()}")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")
    
    # Save collected data
    output_dir = "memorization_analysis_all"
    Path(output_dir).mkdir(exist_ok=True)
    csv_path = Path(output_dir) / "memorization_all_checkpoints.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved data to: {csv_path}")
    
    # Create plots
    print("\nCreating visualizations...")
    create_memorization_plots(df, output_dir)
    
    print("\n" + "="*60)
    print("✓ Analysis complete!")
    print(f"  Results in: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
