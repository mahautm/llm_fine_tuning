import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from paramem.probing.dii import compute_dii_matrix
from paramem.probing.pooled_split import build_pooled_vs_split_summary


def _write_activation_pickle(path: Path, layers_to_arrays: dict[int, np.ndarray]) -> None:
    payload = {layer: [row for row in arr] for layer, arr in layers_to_arrays.items()}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _make_partition_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": "a", "layer": 0, "signal_fraction": 0.25, "num_signal": 4, "num_directions": 16, "n_samples": 64},
            {"dataset": "b", "layer": 0, "signal_fraction": 0.50, "num_signal": 8, "num_directions": 16, "n_samples": 64},
            {"dataset": "a", "layer": 1, "signal_fraction": 0.125, "num_signal": 2, "num_directions": 16, "n_samples": 64},
            {"dataset": "b", "layer": 1, "signal_fraction": 0.375, "num_signal": 6, "num_directions": 16, "n_samples": 64},
        ]
    )


def test_pooled_vs_split_summary(tmp_path: Path):
    rng = np.random.default_rng(0)
    path_a = tmp_path / "a.pickle"
    path_b = tmp_path / "b.pickle"
    _write_activation_pickle(path_a, {0: rng.normal(size=(64, 16)), 1: rng.normal(size=(64, 16))})
    _write_activation_pickle(path_b, {0: rng.normal(size=(64, 16)), 1: rng.normal(size=(64, 16))})

    df = build_pooled_vs_split_summary({"a": path_a, "b": path_b}, _make_partition_df())

    assert list(df.columns) == [
        "layer",
        "pooled_n_samples",
        "pooled_n_features",
        "pooled_sigma2",
        "pooled_lambda_max",
        "pooled_signal_fraction",
        "split_mean_signal_fraction",
        "split_weighted_signal_fraction",
        "split_signal_fraction_std",
        "delta_pooled_minus_split_mean",
        "delta_pooled_minus_split_weighted",
        "num_datasets",
    ]
    assert df.shape[0] == 2
    assert np.isfinite(df["delta_pooled_minus_split_mean"]).all()


def test_dii_matrix_shapes_and_finiteness(tmp_path: Path):
    rng = np.random.default_rng(1)
    path_a = tmp_path / "a.pickle"
    path_b = tmp_path / "b.pickle"
    _write_activation_pickle(path_a, {0: rng.normal(size=(64, 16)), 1: rng.normal(size=(64, 16))})
    _write_activation_pickle(path_b, {0: rng.normal(size=(64, 16)), 1: rng.normal(size=(64, 16))})

    mean_matrices, directional_tables = compute_dii_matrix({"a": path_a, "b": path_b}, _make_partition_df(), k_neighbors=5)

    assert set(mean_matrices.keys()) == {0, 1}
    assert set(directional_tables.keys()) == {0, 1}
    for layer, matrix in mean_matrices.items():
        assert matrix.shape == (2, 2)
        assert np.isfinite(matrix.to_numpy()[np.isfinite(matrix.to_numpy())]).all()
        assert directional_tables[layer].shape[0] == 4
