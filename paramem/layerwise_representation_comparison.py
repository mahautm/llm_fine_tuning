#!/usr/bin/env python3
"""
Experiment 1: Layerwise representation comparison between Full Fine-tuning, LoRA, and Original models.
Uses Information Imbalance (II) and Neighbourhood Overlap to compare all layers.
"""

import numpy as np
import pickle
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import dadapy
import matplotlib.pyplot as plt
import seaborn as sns
import typer
import pandas as pd

app = typer.Typer()


def load_pickle(file_path: str) -> Dict:
    """Load a pickle file containing layer representations."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_quantiles(a: np.ndarray, alphamin: float, alphamax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get quantiles for clipping activations."""
    qmin = np.quantile(a, q=alphamin, axis=1)
    qmax = np.quantile(a, q=alphamax, axis=1)
    return qmin, qmax


def clip_activations(act: np.ndarray, alphamin: float = 0.05, alphamax: float = 0.95) -> np.ndarray:
    """Clip activations to remove outliers."""
    if len(act.shape) == 3:
        reshape = True
        B, T, E = act.shape
        act = np.reshape(act, (B, T*E))
    else:
        reshape = False
    
    qmin, qmax = get_quantiles(act, alphamin, alphamax)
    act = np.clip(act.T, a_min=qmin, a_max=qmax).T
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
        # Get k nearest neighbors for sample i in both representations
        neighbors1 = set(indices1[i, :k])
        neighbors2 = set(indices2[i, :k])
        
        # Compute Jaccard similarity
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        if union > 0:
            jaccard = intersection / union
            overlaps.append(jaccard)
    
    return np.mean(overlaps)


def compare_representations(
    all_activations_A: Dict[str, np.ndarray],
    all_activations_B: Dict[str, np.ndarray],
    model_name_A: str,
    model_name_B: str,
    output_folder: Path,
    limit: int = 2500,
    k_neighbors: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare two sets of layer representations using II and Neighbourhood Overlap.
    
    Args:
        all_activations_A: Dictionary mapping layer IDs to activations (first model)
        all_activations_B: Dictionary mapping layer IDs to activations (second model)
        model_name_A: Name of first model
        model_name_B: Name of second model
        output_folder: Directory to save results
        limit: Maximum number of samples to use
        k_neighbors: Number of neighbors for overlap computation
        
    Returns:
        Tuple of (II results DataFrame, NO results DataFrame)
    """
    # Prepare activations
    all_activations_A = {f"layer_{i}": np.array(all_activations_A[i][:limit]) 
                         for i in all_activations_A.keys()}
    all_activations_B = {f"layer_{i}": np.array(all_activations_B[i][:limit]) 
                         for i in all_activations_B.keys()}
    
    sample_size = all_activations_B["layer_0"].shape[0]
    rows = sorted(all_activations_A.keys())
    cols = sorted(all_activations_B.keys())
    
    ii_ab_results = []
    ii_ba_results = []
    no_results = []
    
    print(f"\n🔄 Comparing {model_name_A} vs {model_name_B}")
    print(f"📊 Sample size: {sample_size}, Layers: {len(rows)} x {len(cols)}")
    
    for row in tqdm(rows, desc="Processing layers (A)"):
        act_A = all_activations_A[row]
        act_A = clip_activations(act_A)
        
        # Compute distances for model A
        dt_A = dadapy.Data(act_A)
        dt_A.compute_distances(maxk=min(sample_size-1, k_neighbors+1), metric="euclidean")
        
        for col in tqdm(cols, desc=f"Comparing with layers (B)", leave=False):
            act_B = all_activations_B[col]
            act_B = clip_activations(act_B)
            
            # Compute distances for model B
            dt_B = dadapy.Data(act_B)
            dt_B.compute_distances(maxk=min(sample_size-1, k_neighbors+1), metric="euclidean")
            
            # Information Imbalance
            ii = dt_A.return_information_imbalace(act_B, subset_size=min(sample_size-1, k_neighbors))
            ii_ab = ii[0][0]
            ii_ba = ii[1][0]
            
            # Neighbourhood Overlap
            no = compute_neighbourhood_overlap(dt_A.dist_indices, dt_B.dist_indices, k=k_neighbors)
            
            tqdm.write(f"{row} <-> {col}: II(A→B)={ii_ab:.4f}, II(B→A)={ii_ba:.4f}, NO={no:.4f}")
            
            ii_ab_results.append({
                "Layer_A": row,
                "Layer_B": col,
                "Model_A": model_name_A,
                "Model_B": model_name_B,
                "II_AB": ii_ab
            })
            
            ii_ba_results.append({
                "Layer_A": row,
                "Layer_B": col,
                "Model_A": model_name_A,
                "Model_B": model_name_B,
                "II_BA": ii_ba
            })
            
            no_results.append({
                "Layer_A": row,
                "Layer_B": col,
                "Model_A": model_name_A,
                "Model_B": model_name_B,
                "Neighbourhood_Overlap": no
            })
    
    # Convert to DataFrames
    df_ii_ab = pd.DataFrame(ii_ab_results)
    df_ii_ba = pd.DataFrame(ii_ba_results)
    df_no = pd.DataFrame(no_results)
    
    # Save results
    output_folder.mkdir(parents=True, exist_ok=True)
    df_ii_ab.to_csv(output_folder / f"II_AB_{model_name_A}_vs_{model_name_B}.csv", index=False)
    df_ii_ba.to_csv(output_folder / f"II_BA_{model_name_A}_vs_{model_name_B}.csv", index=False)
    df_no.to_csv(output_folder / f"NO_{model_name_A}_vs_{model_name_B}.csv", index=False)
    
    return df_ii_ab, df_ii_ba, df_no


def plot_comparison_heatmaps(
    df: pd.DataFrame,
    metric_name: str,
    model_name_A: str,
    model_name_B: str,
    output_file: Path,
    title: Optional[str] = None
):
    """Plot heatmap for layer-wise comparison metrics."""
    # Pivot to create matrix
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
    
    if title is None:
        title = f"{metric_name}: {model_name_A} vs {model_name_B}"
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(f"{model_name_B} Layers", fontsize=12)
    plt.ylabel(f"{model_name_A} Layers", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved heatmap: {output_file}")


def plot_diagonal_comparison(
    results_dict: Dict[str, pd.DataFrame],
    metric_name: str,
    output_file: Path
):
    """Plot diagonal values (same layer comparisons) across different model pairs."""
    plt.figure(figsize=(12, 6))
    
    for comparison_name, df in results_dict.items():
        # Extract diagonal values
        if "II_AB" in df.columns:
            metric_col = "II_AB"
        elif "II_BA" in df.columns:
            metric_col = "II_BA"
        else:
            metric_col = "Neighbourhood_Overlap"
        
        # Filter for diagonal entries only
        diagonal_data = df[df['Layer_A'] == df['Layer_B']].copy()
        
        # Extract layer numbers and create a proper numeric column for sorting
        diagonal_data['layer_num'] = diagonal_data['Layer_A'].apply(lambda x: int(x.split('_')[1]))
        diagonal_data = diagonal_data.sort_values('layer_num')
        
        layer_nums = diagonal_data['layer_num'].values
        values = diagonal_data[metric_col].values
        
        plt.plot(layer_nums, values, marker='o', linewidth=2, markersize=6, label=comparison_name)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} Across Layers (Diagonal Comparison)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved diagonal comparison: {output_file}")


@app.command()
def main(
    models_dir: str = "/home/mmahaut/projects/paramem/models2",
    benchmark_file: str = "/home/mmahaut/projects/paramem/benchmark/dev.jsonl",
    output_dir: str = "/home/mmahaut/projects/paramem/output/layerwise_comparison",
    dataset: str = "pile",  # pile or wikiplus
    limit: int = 2500,
    k_neighbors: int = 50,
    batch_size: int = 16
):
    """
    Compare layerwise representations between Full FT, LoRA FT, and Original models.
    Extracts representations on-the-fly from model checkpoints.
    
    Args:
        models_dir: Directory containing model checkpoints (e.g., models2/)
        benchmark_file: TSV/JSONL file with texts (id\tlabel\ttext)
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
    print(f"\n📥 Loading benchmark data from {benchmark_file}")
    texts, labels = load_benchmark_data(benchmark_file, max_samples=limit)
    print(f"📊 Loaded {len(texts)} samples")
    
    # Find model checkpoints
    models_path = Path(models_dir)
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Map: fsdp-0 = Full FT, fsdp-1 = LoRA
    full_ft_path = None
    lora_ft_path = None
    
    for model_dir in models_path.iterdir():
        if not model_dir.is_dir() or "Llama" not in model_dir.name:
            continue
        if dataset.lower() in model_dir.name.lower():
            if "fsdp-0" in model_dir.name:
                # Find latest checkpoint
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
    
    # Extract representations
    representations = {}
    model_names = {}
    
    # Original model
    print(f"\n{'='*60}")
    print("Extracting from ORIGINAL model")
    print(f"{'='*60}")
    representations['original'] = extract_layer_representations(
        base_model, texts, batch_size=batch_size, is_lora=False
    )
    model_names['original'] = "Original"
    
    # Full FT model
    if full_ft_path:
        print(f"\n{'='*60}")
        print("Extracting from FULL FT model")
        print(f"{'='*60}")
        representations['full_ft'] = extract_layer_representations(
            str(full_ft_path), texts, batch_size=batch_size, is_lora=False
        )
        model_names['full_ft'] = f"Full-FT-{dataset}"
    
    # LoRA FT model
    if lora_ft_path:
        print(f"\n{'='*60}")
        print("Extracting from LORA FT model")
        print(f"{'='*60}")
        # Check if LoRA adapter exists (try both .bin and .safetensors)
        has_adapter = (lora_ft_path / "adapter_model.safetensors").exists() or (lora_ft_path / "adapter_model.bin").exists()
        
        if has_adapter:
            representations['lora_ft'] = extract_layer_representations(
                str(lora_ft_path), texts, batch_size=batch_size, is_lora=True, base_model_name=base_model
            )
            model_names['lora_ft'] = f"LoRA-FT-{dataset}"
        else:
            print(f"⚠️  No LoRA adapter found at {lora_ft_path}")
    
    if len(representations) < 2:
        print("❌ Error: Need at least 2 models to compare!")
        return
    
    # Perform all pairwise comparisons
    comparisons = []
    all_ii_ab = {}
    all_ii_ba = {}
    all_no = {}
    
    model_types = list(representations.keys())
    for i in range(len(model_types)):
        for j in range(i + 1, len(model_types)):
            type_a = model_types[i]
            type_b = model_types[j]
            
            print(f"\n{'='*60}")
            print(f"Comparison {len(comparisons)+1}: {type_a} vs {type_b}")
            print(f"{'='*60}")
            
            df_ii_ab, df_ii_ba, df_no = compare_representations(
                representations[type_a],
                representations[type_b],
                model_names[type_a],
                model_names[type_b],
                output_path,
                limit=limit,
                k_neighbors=k_neighbors
            )
            
            comparison_name = f"{type_a}_vs_{type_b}"
            all_ii_ab[comparison_name] = df_ii_ab
            all_ii_ba[comparison_name] = df_ii_ba
            all_no[comparison_name] = df_no
            
            # Plot heatmaps
            plot_comparison_heatmaps(
                df_ii_ab,
                "Information Imbalance (A→B)",
                model_names[type_a],
                model_names[type_b],
                output_path / f"heatmap_II_AB_{comparison_name}.png"
            )
            
            plot_comparison_heatmaps(
                df_ii_ba,
                "Information Imbalance (B→A)",
                model_names[type_a],
                model_names[type_b],
                output_path / f"heatmap_II_BA_{comparison_name}.png"
            )
            
            plot_comparison_heatmaps(
                df_no,
                "Neighbourhood Overlap",
                model_names[type_a],
                model_names[type_b],
                output_path / f"heatmap_NO_{comparison_name}.png"
            )
    
    # Plot diagonal comparisons across all model pairs
    if all_ii_ab:
        plot_diagonal_comparison(
            all_ii_ab,
            "Information Imbalance (A→B)",
            output_path / f"diagonal_II_AB_comparison_{dataset}.png"
        )
    
    if all_ii_ba:
        plot_diagonal_comparison(
            all_ii_ba,
            "Information Imbalance (B→A)",
            output_path / f"diagonal_II_BA_comparison_{dataset}.png"
        )
    
    if all_no:
        plot_diagonal_comparison(
            all_no,
            "Neighbourhood Overlap",
            output_path / f"diagonal_NO_comparison_{dataset}.png"
        )
    
    print(f"\n{'='*60}")
    print(f"✅ All comparisons completed!")
    print(f"📁 Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    app()
