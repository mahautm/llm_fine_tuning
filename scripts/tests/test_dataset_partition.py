import pickle
from pathlib import Path

import numpy as np

from paramem.spectral.dataset_partition import DatasetPartition


def _write_activation_pickle(path: Path, layers_to_arrays: dict[int, np.ndarray]) -> None:
    payload = {layer: [row for row in arr] for layer, arr in layers_to_arrays.items()}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_dataset_partition_compute_and_save(tmp_path: Path):
    rng = np.random.default_rng(0)

    ds_a = {
        0: rng.normal(size=(256, 32)),
        1: rng.normal(size=(256, 32)),
    }
    ds_b = {
        0: rng.normal(size=(256, 32)),
        1: rng.normal(size=(256, 32)),
    }

    path_a = tmp_path / "a.pickle"
    path_b = tmp_path / "b.pickle"
    _write_activation_pickle(path_a, ds_a)
    _write_activation_pickle(path_b, ds_b)

    runner = DatasetPartition({"a": path_a, "b": path_b}, layers=[0, 1])
    result = runner.compute_all_partitions()

    assert result.partition_tensor.ndim == 3
    assert result.partition_tensor.shape[0] == 2
    assert result.partition_tensor.shape[1] == 2
    assert result.lambda_max_matrix.shape == (2, 2)
    assert result.signal_fraction_matrix.shape == (2, 2)

    overlap = DatasetPartition.compare_partitions(
        result.partition_tensor[0, 0], result.partition_tensor[1, 0]
    )
    assert 0.0 <= overlap <= 1.0

    npz_path, json_path = DatasetPartition.save_result(result, tmp_path / "partition_result")
    assert npz_path.exists()
    assert json_path.exists()
