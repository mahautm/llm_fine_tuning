from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .activation_cov import compute_cov, compute_eigenspectrum
from .mp_fit import fit_mp, partition


class DatasetPartitionResult(NamedTuple):
    datasets: List[str]
    layers: List[int]
    direction_count: int
    partition_tensor: np.ndarray
    lambda_max_matrix: np.ndarray
    signal_fraction_matrix: np.ndarray


class DatasetPartition:
    """
    Build dataset-conditional MP partitions from activation pickles.

    Each input pickle is expected to be Dict[layer_id -> List[np.ndarray]]
    compatible with output from `paramem/extract_hidden.py`.
    """

    def __init__(
        self,
        dataset_to_pickle: Dict[str, str | Path],
        layers: Optional[Sequence[int]] = None,
    ) -> None:
        if not dataset_to_pickle:
            raise ValueError("dataset_to_pickle cannot be empty.")

        self.dataset_to_pickle: Dict[str, Path] = {
            name: Path(path).expanduser().resolve() for name, path in dataset_to_pickle.items()
        }
        self._user_layers = None if layers is None else sorted(set(int(l) for l in layers))

        for name, path in self.dataset_to_pickle.items():
            if not path.exists():
                raise FileNotFoundError(f"Dataset '{name}' pickle not found: {path}")

    @staticmethod
    def _load_pickle(path: Path) -> Dict[int, np.ndarray]:
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        out: Dict[int, np.ndarray] = {}
        for layer, vectors in payload.items():
            arr = np.asarray(vectors, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"Layer {layer} in {path} is not 2D [n_samples, n_features].")
            out[int(layer)] = arr
        return out

    def _resolve_layers(self, all_layer_maps: Dict[str, Dict[int, np.ndarray]]) -> List[int]:
        common_layers = None
        for _, layer_map in all_layer_maps.items():
            layer_set = set(layer_map.keys())
            common_layers = layer_set if common_layers is None else (common_layers & layer_set)

        if not common_layers:
            raise RuntimeError("No common layers across datasets.")

        if self._user_layers is None:
            return sorted(common_layers)

        selected = [layer for layer in self._user_layers if layer in common_layers]
        if not selected:
            raise RuntimeError("None of the requested layers are common across datasets.")
        return selected

    @staticmethod
    def _align_direction_count(arrays: Iterable[np.ndarray]) -> int:
        dims = [arr.shape[1] for arr in arrays]
        return int(min(dims))

    def compute_all_partitions(self) -> DatasetPartitionResult:
        loaded = {name: self._load_pickle(path) for name, path in self.dataset_to_pickle.items()}
        datasets = sorted(loaded.keys())
        layers = self._resolve_layers(loaded)

        min_directions_per_layer = {
            layer: self._align_direction_count(loaded[d][layer] for d in datasets) for layer in layers
        }
        direction_count = min(min_directions_per_layer.values())

        partition_tensor = np.zeros((len(datasets), len(layers), direction_count), dtype=bool)
        lambda_max_matrix = np.zeros((len(datasets), len(layers)), dtype=np.float64)
        signal_fraction_matrix = np.zeros((len(datasets), len(layers)), dtype=np.float64)

        for di, dataset in enumerate(datasets):
            for li, layer in enumerate(layers):
                activations = loaded[dataset][layer]
                n_samples, n_features = activations.shape

                cov = compute_cov(activations, center=True)
                evals = compute_eigenspectrum(cov, descending=False)

                evals = evals[-direction_count:]
                gamma_eff = min(n_samples, n_features) / max(n_samples, n_features)
                mp = fit_mp(evals, gamma=gamma_eff)
                signal_mask, _ = partition(evals, mp.lambda_max)

                partition_tensor[di, li, :] = signal_mask
                lambda_max_matrix[di, li] = mp.lambda_max
                signal_fraction_matrix[di, li] = float(signal_mask.mean())

        return DatasetPartitionResult(
            datasets=datasets,
            layers=layers,
            direction_count=direction_count,
            partition_tensor=partition_tensor,
            lambda_max_matrix=lambda_max_matrix,
            signal_fraction_matrix=signal_fraction_matrix,
        )

    @staticmethod
    def compare_partitions(
        partition_a: np.ndarray,
        partition_b: np.ndarray,
    ) -> float:
        """
        Jaccard overlap between signal direction sets.
        """
        a = np.asarray(partition_a, dtype=bool)
        b = np.asarray(partition_b, dtype=bool)
        if a.shape != b.shape:
            raise ValueError("partition_a and partition_b must have same shape.")

        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 1.0
        return float(inter / union)

    @staticmethod
    def save_result(result: DatasetPartitionResult, output_prefix: str | Path) -> Tuple[Path, Path]:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        npz_path = output_prefix.with_suffix(".npz")
        json_path = output_prefix.with_suffix(".json")

        np.savez_compressed(
            npz_path,
            partition_tensor=result.partition_tensor,
            lambda_max_matrix=result.lambda_max_matrix,
            signal_fraction_matrix=result.signal_fraction_matrix,
            datasets=np.array(result.datasets, dtype=object),
            layers=np.array(result.layers, dtype=np.int64),
            direction_count=np.array([result.direction_count], dtype=np.int64),
        )

        metadata = {
            "datasets": result.datasets,
            "layers": result.layers,
            "direction_count": result.direction_count,
            "shape": list(result.partition_tensor.shape),
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return npz_path, json_path
