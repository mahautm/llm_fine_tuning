#!/usr/bin/env python3
"""
Run MP partition stability metrics from cached activation pickles.

Expected pickle format (compatible with paramem/extract_hidden.py):
- Dict[int, List[np.ndarray]] mapping layer_id -> per-sample activation vectors.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import typer

from paramem.spectral.activation_cov import compute_cov, compute_eigenspectrum
from paramem.spectral.mp_fit import fit_mp, partition

app = typer.Typer(add_completion=False)


def _load_layer_activations(path: Path) -> Dict[int, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)

    out: Dict[int, np.ndarray] = {}
    for layer_id, vectors in data.items():
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Layer {layer_id} in {path} is not 2D [n_samples, n_features].")
        out[int(layer_id)] = arr
    return out


def _parse_input_spec(input_specs: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for spec in input_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --input format '{spec}'. Use dataset_name=/absolute/or/relative/path.pickle"
            )
        dataset_name, raw_path = spec.split("=", 1)
        dataset_name = dataset_name.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not dataset_name:
            raise ValueError(f"Dataset name is empty in spec '{spec}'.")
        if not path.exists():
            raise FileNotFoundError(f"Input pickle not found: {path}")
        mapping[dataset_name] = path
    if not mapping:
        raise ValueError("No inputs were provided.")
    return mapping


@app.command()
def main(
    input: List[str] = typer.Option(
        ..., "--input", help="Dataset spec: dataset_name=/path/to/activations.pickle. Repeat per dataset."
    ),
    layers: str = typer.Option(
        "", help="Comma-separated layers to evaluate. Empty means all layers found in each file."
    ),
    output_dir: str = typer.Option(
        "results/mp_reservoir/spectra", help="Output directory for CSV metrics."
    ),
    tag: str = typer.Option("exp1", help="Tag used in output filenames."),
) -> None:
    input_map = _parse_input_spec(input)
    selected_layers = None
    if layers.strip():
        selected_layers = {int(x.strip()) for x in layers.split(",") if x.strip()}

    rows = []

    for dataset_name, pickle_path in input_map.items():
        layer_acts = _load_layer_activations(pickle_path)
        layer_ids = sorted(layer_acts.keys())

        for layer_id in layer_ids:
            if selected_layers is not None and layer_id not in selected_layers:
                continue

            activations = layer_acts[layer_id]
            n_samples, n_features = activations.shape
            if n_samples < 3:
                continue

            cov = compute_cov(activations, center=True)
            evals = compute_eigenspectrum(cov, descending=False)

            gamma_eff = min(n_samples, n_features) / max(n_samples, n_features)
            mp = fit_mp(evals, gamma=gamma_eff)
            signal_mask, _ = partition(evals, mp.lambda_max)

            signal_fraction = float(np.mean(signal_mask))
            rows.append(
                {
                    "dataset": dataset_name,
                    "layer": layer_id,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "gamma_eff": gamma_eff,
                    "sigma2": mp.sigma2,
                    "lambda_min": mp.lambda_min,
                    "lambda_max": mp.lambda_max,
                    "signal_fraction": signal_fraction,
                    "num_signal": int(signal_mask.sum()),
                    "num_directions": int(signal_mask.size),
                    "input_pickle": str(pickle_path),
                }
            )

    if not rows:
        raise RuntimeError("No rows were produced. Check your layer selection and input pickles.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(["dataset", "layer"])
    out_csv = out_dir / f"mp_partition_metrics_{tag}.csv"
    df.to_csv(out_csv, index=False)

    pivot = df.pivot(index="layer", columns="dataset", values="signal_fraction")
    out_pivot = out_dir / f"mp_signal_fraction_pivot_{tag}.csv"
    pivot.to_csv(out_pivot)

    print(f"Saved metrics to {out_csv}")
    print(f"Saved pivot to   {out_pivot}")


if __name__ == "__main__":
    app()
