#!/usr/bin/env python3
"""
Self-comparison: Compare each model against itself across all layers.
Computes Information Imbalance and Neighbourhood Overlap for intra-model layer relationships.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
import dadapy

app = typer.Typer()


def clip_activations(act: np.ndarray, alphamin: float = 0.05, alphamax: float = 0.95) -> np.ndarray:
    """Clip activations to remove outliers."""
    qmin = np.quantile(act, q=alphamin, axis=0)
    qmax = np.quantile(act, q=alphamax, axis=0)
    act = np.clip(act, a_min=qmin, a_max=qmax)
    return act


def compute_neighbourhood_overlap(indices1: np.ndarray, indices2: np.ndarray, k: int = 50) -> float:
    """
    Compute neighbourhood overlap between two sets of k-nearest neighbor indices.
    
    Args:
        indices1: KNN indices from first representation (N x k)
        indices2: KNN indices from second representation (N x k)
        k: Number of neighbors to consider
        
    Returns:
        Average Jaccard similarity across all samples
    """
    N = indices1.shape[0]
    overlaps = []
    
    for i in range(N):
        neighbors1 = set(indices1[i, :k])
        neighbors2 = set(indices2[i, :k])
        
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        if union > 0:
            jaccard = intersection / union
            overlaps.append(jaccard)
    
    return np.mean(overlaps)


def compare_representations(act_a: np.ndarray, act_b: np.ndarray, k: int = 50) -> Tuple[float, float, float]:
    """
    Compare two layer representations using II and NO.
    
    Args:
        act_a: First layer activations (N x D)
        act_b: Second layer activations (N x D)
        k: Number of neighbors
        
    Returns:
        Tuple of (II_AB, II_BA, NO)
    """
    # Ensure we have numpy arrays
    if not isinstance(act_a, np.ndarray):
        act_a = np.array(act_a)
    if not isinstance(act_b, np.ndarray):
        act_b = np.array(act_b)
    
    # Clip activations and ensure numeric dtype/contiguous layout
    act_a_clipped = clip_activations(act_a)
    act_b_clipped = clip_activations(act_b)

    # dadapy / sklearn expect regular numeric ndarrays (float32 or float64)
    act_a_clipped = np.ascontiguousarray(act_a_clipped.astype(np.float32))
    act_b_clipped = np.ascontiguousarray(act_b_clipped.astype(np.float32))
    
    # Compute data objects for each representation
    data_a = dadapy.Data(act_a_clipped)
    data_a.compute_distances(maxk=min(act_a_clipped.shape[0]-1, k+1), metric="euclidean")
    knn_indices_a = data_a.dist_indices[:, :k]

    data_b = dadapy.Data(act_b_clipped)
    data_b.compute_distances(maxk=min(act_b_clipped.shape[0]-1, k+1), metric="euclidean")
    knn_indices_b = data_b.dist_indices[:, :k]
    
    # Information Imbalance - pass numpy arrays to the method
    ii = data_a.return_information_imbalace(act_b_clipped, subset_size=min(act_a_clipped.shape[0]-1, k))
    ii_ab = ii[0][0]
    ii_ba = ii[1][0]
    
    # Neighbourhood Overlap
    no = compute_neighbourhood_overlap(knn_indices_a, knn_indices_b, k=k)
    
    return ii_ab, ii_ba, no


@app.command()
def main(
    models_dir: str = "/home/mmahaut/projects/paramem/models2",
    benchmark_file: str = "/home/mmahaut/projects/paramem/benchmark/dev.jsonl",
    output_dir: str = "/home/mmahaut/projects/paramem/output/layerwise_self_comparison",
    dataset: str = "pile",  # pile or wikiplus
    limit: int = 1000,
    k_neighbors: int = 50,
    batch_size: int = 8
):
    """
    Compare each model's layers against itself to analyze internal representation evolution.
    
    Args:
        models_dir: Directory containing model checkpoints
        benchmark_file: JSONL file with texts
        output_dir: Directory to save results
        dataset: Dataset to use (pile or wikiplus)
        limit: Maximum number of samples
        k_neighbors: Number of neighbors for overlap computation
        batch_size: Batch size for extraction
    """
    from extract_representations_utils import extract_layer_representations, load_benchmark_data
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load benchmark data
    texts, _ = load_benchmark_data(benchmark_file, max_samples=limit)
    print(f"📥 Loaded {len(texts)} texts from {benchmark_file}")
    
    # Find model checkpoints
    models_path = Path(models_dir)
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    full_ft_path = None
    lora_ft_path = None
    
    for model_dir in models_path.iterdir():
        if not model_dir.is_dir() or "Llama" not in model_dir.name:
            continue
        if dataset.lower() in model_dir.name.lower():
            if "fsdp-0" in model_dir.name:
                checkpoints = sorted([d for d in model_dir.iterdir() if d.name.startswith("checkpoint-")])
                if checkpoints:
                    full_ft_path = checkpoints[-1]
            elif "fsdp-1" in model_dir.name:
                checkpoints = sorted([d for d in model_dir.iterdir() if d.name.startswith("checkpoint-")])
                if checkpoints:
                    lora_ft_path = checkpoints[-1]
    
    print(f"\n🔍 Found model paths for {dataset}:")
    print(f"  Original: {base_model}")
    print(f"  Full FT: {full_ft_path}")
    print(f"  LoRA FT: {lora_ft_path}")
    
    # Extract representations for each model
    models_to_process = {
        'Original': (base_model, False, None),
        f'Full-FT-{dataset}': (str(full_ft_path), False, None) if full_ft_path else None,
        f'LoRA-FT-{dataset}': (str(lora_ft_path), True, base_model) if lora_ft_path else None,
    }
    
    for model_name, model_info in models_to_process.items():
        if model_info is None:
            print(f"⚠️  Skipping {model_name} (not found)")
            continue
        
        model_path, is_lora, base_model_name = model_info
        
        print(f"\n{'='*60}")
        print(f"🔬 Processing {model_name}")
        print(f"{'='*60}")
        
        # Extract representations
        representations = extract_layer_representations(
            model_path, texts, batch_size=batch_size, 
            is_lora=is_lora, base_model_name=base_model_name
        )
        
        num_layers = len(representations)
        print(f"📊 Extracted {num_layers} layers with {len(texts)} samples each")
        
        # Self-comparison: compare all layer pairs within this model
        print(f"\n🔄 Computing self-comparison metrics...")
        
        results_ii_ab = []
        results_ii_ba = []
        results_no = []
        
        for i in tqdm(range(num_layers), desc="Layer pairs"):
            layer_i_name = f"layer_{i}"
            
            for j in range(num_layers):
                layer_j_name = f"layer_{j}"
                
                # Convert lists to numpy arrays
                act_i = np.array(representations[i])
                act_j = np.array(representations[j])
                
                # Compute metrics
                ii_ab, ii_ba, no = compare_representations(
                    act_i, 
                    act_j,
                    k=k_neighbors
                )
                
                results_ii_ab.append({
                    'Layer_A': layer_i_name,
                    'Layer_B': layer_j_name,
                    'II_AB': ii_ab
                })
                results_ii_ba.append({
                    'Layer_A': layer_i_name,
                    'Layer_B': layer_j_name,
                    'II_BA': ii_ba
                })
                results_no.append({
                    'Layer_A': layer_i_name,
                    'Layer_B': layer_j_name,
                    'Neighbourhood_Overlap': no
                })
        
        # Save results
        model_output_dir = output_path / dataset / model_name.replace('/', '_').replace(' ', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        df_ii_ab = pd.DataFrame(results_ii_ab)
        df_ii_ba = pd.DataFrame(results_ii_ba)
        df_no = pd.DataFrame(results_no)
        
        df_ii_ab.to_csv(model_output_dir / "self_comparison_II_AB.csv", index=False)
        df_ii_ba.to_csv(model_output_dir / "self_comparison_II_BA.csv", index=False)
        df_no.to_csv(model_output_dir / "self_comparison_NO.csv", index=False)
        
        # Plot heatmaps
        plot_self_comparison_heatmap(
            df_ii_ab, "Information Imbalance (A→B)", model_name,
            model_output_dir / "heatmap_self_II_AB.png"
        )
        plot_self_comparison_heatmap(
            df_ii_ba, "Information Imbalance (B→A)", model_name,
            model_output_dir / "heatmap_self_II_BA.png"
        )
        plot_self_comparison_heatmap(
            df_no, "Neighbourhood Overlap", model_name,
            model_output_dir / "heatmap_self_NO.png"
        )
        
        print(f"✅ Saved self-comparison results for {model_name} to {model_output_dir}")
    
    print(f"\n{'='*60}")
    print("✅ All self-comparisons completed!")
    print(f"📁 Results saved to: {output_path}")
    print(f"{'='*60}")


def plot_self_comparison_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    model_name: str,
    output_file: Path
):
    """Plot a heatmap of self-comparison metrics."""
    # Determine metric column
    if "II_AB" in df.columns:
        metric_col = "II_AB"
    elif "II_BA" in df.columns:
        metric_col = "II_BA"
    else:
        metric_col = "Neighbourhood_Overlap"
    
    matrix = df.pivot(index="Layer_A", columns="Layer_B", values=metric_col)
    
    # Extract layer numbers for proper sorting
    def extract_layer_num(layer_str):
        return int(layer_str.split('_')[1])
    
    sorted_layers_A = sorted(matrix.index, key=extract_layer_num)
    sorted_layers_B = sorted(matrix.columns, key=extract_layer_num)
    matrix = matrix.reindex(index=sorted_layers_A, columns=sorted_layers_B)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Choose colormap based on metric
    if "Overlap" in metric_name:
        cmap = "viridis"  # Higher is more similar
    else:
        cmap = "coolwarm"  # II interpretation varies
    
    sns.heatmap(matrix, annot=False, cmap=cmap, cbar_kws={'label': metric_name})
    
    plt.title(f"{metric_name}: {model_name} Self-Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Layer", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved heatmap: {output_file}")


if __name__ == "__main__":
    app()
