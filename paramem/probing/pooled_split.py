from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from paramem.spectral.activation_cov import compute_cov, compute_eigendecomposition
from paramem.spectral.mp_fit import fit_mp, partition


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


def load_partition_metrics(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty metrics CSV: {path}")
    required = {"dataset", "layer", "signal_fraction", "num_signal", "num_directions"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def _select_layers(layer_maps: Iterable[Dict[int, np.ndarray]]) -> List[int]:
    common = None
    for layer_map in layer_maps:
        current = set(layer_map.keys())
        common = current if common is None else (common & current)
    if not common:
        raise RuntimeError("No common layers across activation pickles.")
    return sorted(common)


def build_pooled_vs_split_summary(
    dataset_to_pickle: Dict[str, str | Path],
    partition_metrics: pd.DataFrame,
    layers: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Compare pooled activations against split dataset activations.

    pooled_* columns are computed from concatenated activations across datasets.
    split_* columns summarize the per-dataset MP partitions from the input metrics table.
    """
    if not dataset_to_pickle:
        raise ValueError("dataset_to_pickle cannot be empty.")

    loaded = {name: load_layer_activations(path) for name, path in dataset_to_pickle.items()}
    selected_layers = sorted(set(int(layer) for layer in layers)) if layers is not None else _select_layers(loaded.values())

    rows = []
    for layer in selected_layers:
        layer_activations = []
        layer_lengths = []
        layer_features = None

        for dataset_name, layer_map in loaded.items():
            if layer not in layer_map:
                continue
            activations = layer_map[layer]
            layer_activations.append((dataset_name, activations))
            layer_lengths.append(activations.shape[0])
            layer_features = activations.shape[1]

        if not layer_activations or layer_features is None:
            continue

        pooled_acts = np.concatenate([acts for _, acts in layer_activations], axis=0)
        pooled_cov = compute_cov(pooled_acts, center=True)
        pooled_evals = np.linalg.eigvalsh(pooled_cov)
        gamma_eff = min(pooled_acts.shape[0], pooled_acts.shape[1]) / max(pooled_acts.shape[0], pooled_acts.shape[1])
        pooled_fit = fit_mp(pooled_evals, gamma=gamma_eff)
        pooled_signal_mask, _ = partition(pooled_evals, pooled_fit.lambda_max)

        split_rows = partition_metrics.loc[partition_metrics["layer"] == layer].copy()
        if split_rows.empty:
            continue

        split_signal_fraction = float(split_rows["signal_fraction"].mean())
        split_signal_fraction_std = float(split_rows["signal_fraction"].std(ddof=0)) if len(split_rows) > 1 else 0.0
        sample_weights = split_rows["n_samples"] if "n_samples" in split_rows.columns else None
        if sample_weights is not None:
            split_weighted_fraction = float(np.average(split_rows["signal_fraction"], weights=split_rows["n_samples"]))
        else:
            split_weighted_fraction = split_signal_fraction

        rows.append(
            {
                "layer": layer,
                "pooled_n_samples": int(pooled_acts.shape[0]),
                "pooled_n_features": int(pooled_acts.shape[1]),
                "pooled_sigma2": pooled_fit.sigma2,
                "pooled_lambda_max": pooled_fit.lambda_max,
                "pooled_signal_fraction": float(np.mean(pooled_signal_mask)),
                "split_mean_signal_fraction": split_signal_fraction,
                "split_weighted_signal_fraction": split_weighted_fraction,
                "split_signal_fraction_std": split_signal_fraction_std,
                "delta_pooled_minus_split_mean": float(np.mean(pooled_signal_mask) - split_signal_fraction),
                "delta_pooled_minus_split_weighted": float(np.mean(pooled_signal_mask) - split_weighted_fraction),
                "num_datasets": int(len(split_rows)),
            }
        )

    if not rows:
        raise RuntimeError("No pooled-vs-split rows were produced.")

    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
