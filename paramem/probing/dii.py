from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from paramem.spectral.activation_cov import compute_cov, compute_eigendecomposition, project_activations

try:
    import dadapy
except Exception as exc:  # pragma: no cover - dependency is expected in the project env
    dadapy = None
    _DADAPY_IMPORT_ERROR = exc
else:
    _DADAPY_IMPORT_ERROR = None


def load_layer_activations(pickle_path: str | Path) -> Dict[int, np.ndarray]:
    path = Path(pickle_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    loaded: Dict[int, np.ndarray] = {}
    for layer, vectors in payload.items():
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Layer {layer} in {path} is not 2D [n_samples, n_features].")
        loaded[int(layer)] = arr
    return loaded


def _align_rows(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(left.shape[0], right.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 rows to compare embeddings.")
    return left[:n], right[:n]


def information_imbalance(x: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """
    Compute information imbalance using dadapy when available.
    Falls back to a nearest-neighbor rank approximation if dadapy is unavailable.
    """
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")
    if left.shape[0] != right.shape[0]:
        left, right = _align_rows(left, right)

    n = left.shape[0]
    if n <= 2:
        raise ValueError("Need at least 3 samples for information imbalance.")

    subset_size = min(max(1, int(k)), n - 1)

    if dadapy is not None:
        data = dadapy.Data(left)
        data.compute_distances(maxk=n - 1, metric="euclidean")
        imbalance = data.return_information_imbalace(right, subset_size=subset_size)
        return float(imbalance[0][0])

    # Fallback approximation based on nearest-neighbor rank overlap.
    nn_left = NearestNeighbors(n_neighbors=subset_size + 1, metric="euclidean").fit(left)
    left_neighbors = nn_left.kneighbors(return_distance=False)[:, 1:]
    nn_right = NearestNeighbors(n_neighbors=n - 1, metric="euclidean").fit(right)
    right_neighbors = nn_right.kneighbors(return_distance=False)

    ranks = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        ranks[i, right_neighbors[i]] = np.arange(1, n)
    avg_rank = float(np.mean([np.mean(ranks[i, nbrs]) for i, nbrs in enumerate(left_neighbors)]))
    return avg_rank / max(n - 1, 1)


def compute_signal_subspace_embedding(
    activations: np.ndarray,
    signal_components: int,
    center: bool = True,
    max_samples: Optional[int] = None,
    random_state: int = 0,
) -> np.ndarray:
    """
    Project activations into the top signal subspace inferred from MP counts.
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("activations must be 2D.")
    if signal_components <= 0:
        signal_components = 1

    if max_samples is not None and x.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))
        x = x[indices]

    cov = compute_cov(x, center=center)
    _, evecs = compute_eigendecomposition(cov, descending=True)
    basis = evecs[:, : min(signal_components, evecs.shape[1])]
    return project_activations(x, basis, center=center)


def compute_dii_matrix(
    dataset_to_pickle: Dict[str, str | Path],
    partition_metrics: pd.DataFrame,
    layers: Optional[Iterable[int]] = None,
    k_neighbors: int = 10,
    max_samples: int = 2000,
) -> tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Build per-layer pairwise DII matrices between dataset signal subspaces.

    Returns:
        (mean_matrices, directional_matrices)
        - mean_matrices[layer]: symmetric matrix of average II(left->right, right->left)
        - directional_matrices[layer]: long-form table with ab and ba values
    """
    if not dataset_to_pickle:
        raise ValueError("dataset_to_pickle cannot be empty.")

    loaded = {name: load_layer_activations(path) for name, path in dataset_to_pickle.items()}
    datasets = sorted(loaded.keys())
    selected_layers = sorted(set(int(layer) for layer in layers)) if layers is not None else sorted(
        set.intersection(*(set(layer_map.keys()) for layer_map in loaded.values()))
    )

    mean_matrices: Dict[int, pd.DataFrame] = {}
    directional_tables: Dict[int, pd.DataFrame] = {}

    for layer in selected_layers:
        layer_rows = partition_metrics.loc[partition_metrics["layer"] == layer].copy()
        if layer_rows.empty:
            continue

        signal_counts = {
            str(row["dataset"]): max(int(row["num_signal"]), 1)
            for _, row in layer_rows.iterrows()
            if str(row["dataset"]) in loaded
        }
        if not signal_counts:
            continue

        pairwise = pd.DataFrame(np.nan, index=datasets, columns=datasets, dtype=float)
        records = []

        embeddings = {}
        for dataset in datasets:
            if layer not in loaded[dataset]:
                continue
            embeddings[dataset] = compute_signal_subspace_embedding(
                loaded[dataset][layer],
                signal_components=signal_counts.get(dataset, 1),
                    max_samples=max_samples,
                    random_state=layer + len(dataset),
            )

        for left in datasets:
            for right in datasets:
                if left not in embeddings or right not in embeddings:
                    continue
                emb_left, emb_right = _align_rows(embeddings[left], embeddings[right])
                k_eff = min(k_neighbors, emb_left.shape[0] - 1)
                if k_eff < 1:
                    continue
                ii_lr = information_imbalance(emb_left, emb_right, k=k_eff)
                ii_rl = information_imbalance(emb_right, emb_left, k=k_eff)
                ii_mean = float((ii_lr + ii_rl) / 2.0)
                pairwise.loc[left, right] = ii_mean
                records.append(
                    {
                        "layer": layer,
                        "dataset_left": left,
                        "dataset_right": right,
                        "ii_left_to_right": float(ii_lr),
                        "ii_right_to_left": float(ii_rl),
                        "ii_mean": ii_mean,
                        "signal_count_left": int(signal_counts.get(left, 1)),
                        "signal_count_right": int(signal_counts.get(right, 1)),
                    }
                )

        mean_matrices[layer] = pairwise
        directional_tables[layer] = pd.DataFrame.from_records(records)

    if not mean_matrices:
        raise RuntimeError("No DII matrices were produced.")

    return mean_matrices, directional_tables
