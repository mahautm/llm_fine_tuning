#!/usr/bin/env python3
"""
Analyze checkpoint-level MP dynamics for FT vs LoRA runs.

This script computes per-checkpoint, per-layer MP quantities from activation pickles,
then compares informative-dimension masks against a baseline checkpoint to quantify
"noise -> informative" and "informative -> informative" dynamics.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from paramem.spectral.activation_cov import compute_cov, compute_eigenspectrum
from paramem.spectral.mp_fit import fit_mp, partition

app = typer.Typer(add_completion=False)


@dataclass
class ModelMeta:
    model_dir: Path
    model_name: str
    training_type: str
    dataset: str


def _parse_model_meta(model_dir: Path) -> ModelMeta:
    name = model_dir.name
    training_type = "unknown"
    dataset = "unknown"

    if "fsdp-1" in name:
        training_type = "LoRA"
    elif "fsdp-0" in name:
        training_type = "Full-FT"

    lower = name.lower()
    if "pile" in lower:
        dataset = "pile"
    elif "wikiplus" in lower or "wikidata" in lower:
        dataset = "wikidata"
    elif "mmlu" in lower:
        dataset = "mmlu"

    return ModelMeta(
        model_dir=model_dir,
        model_name=name,
        training_type=training_type,
        dataset=dataset,
    )


def _checkpoint_num(path: Path) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)", str(path))
    if m is None:
        return None
    return int(m.group(1))


def _parse_layers(layers: str) -> Optional[List[int]]:
    if not layers.strip():
        return None
    return sorted({int(x.strip()) for x in layers.split(",") if x.strip()})


def _load_pickle(path: Path) -> Dict[int, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    out: Dict[int, np.ndarray] = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected pickle format in {path}")

    for layer, vectors in payload.items():
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2:
            continue
        out[int(layer)] = arr
    return out


def _compute_layer_metrics(activations: np.ndarray) -> Tuple[dict, np.ndarray]:
    n_samples, n_features = activations.shape
    centered = activations - activations.mean(axis=0, keepdims=True)

    cov = compute_cov(activations, center=True)
    evals = compute_eigenspectrum(cov, descending=False)

    gamma_eff = min(n_samples, n_features) / max(n_samples, n_features)
    mp = fit_mp(evals, gamma=gamma_eff)
    eig_signal_mask, _ = partition(evals, mp.lambda_max)

    # Dimension-level informative mask in activation basis.
    feature_vars = centered.var(axis=0, ddof=1)
    informative_mask = feature_vars > mp.lambda_max

    row = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "gamma_eff": float(gamma_eff),
        "sigma2": float(mp.sigma2),
        "lambda_min": float(mp.lambda_min),
        "lambda_max": float(mp.lambda_max),
        "signal_fraction_eig": float(eig_signal_mask.mean()),
        "num_signal_eig": int(eig_signal_mask.sum()),
        "signal_fraction_feature": float(informative_mask.mean()),
        "num_informative_feature": int(informative_mask.sum()),
    }
    return row, informative_mask.astype(bool)


def _find_model_dirs(roots: List[Path], model_glob: str) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend([p for p in root.glob(model_glob) if p.is_dir()])
    return sorted(set(out))


@app.command()
def main(
    model_root: List[str] = typer.Option(
        [
            "/home/mmahaut/projects/paramem/models3",
            "/home/mmahaut/projects/paramem/models2",
        ],
        "--model-root",
        help="Root directory containing model run folders. Repeatable.",
    ),
    model_glob: str = typer.Option("*fsdp-*", help="Glob for model run folders under each root."),
    activation_glob: str = typer.Option(
        "**/*.act.pickle",
        help="Glob under each checkpoint directory for activation pickles.",
    ),
    layers: str = typer.Option("", help="Comma-separated layer IDs to keep. Empty = all."),
    baseline: str = typer.Option(
        "first",
        help="Baseline checkpoint policy: 'first' (minimum checkpoint number).",
    ),
    output_dir: str = typer.Option(
        "results/mp_reservoir/ft_lora_dynamics",
        help="Output directory for CSV artifacts.",
    ),
    tag: str = typer.Option("ft_lora_mp_dynamics", help="Tag for output file names."),
) -> None:
    if baseline != "first":
        raise ValueError("Only baseline='first' is currently supported.")

    selected_layers = _parse_layers(layers)
    roots = [Path(p).expanduser().resolve() for p in model_root]
    model_dirs = _find_model_dirs(roots, model_glob)

    if not model_dirs:
        raise RuntimeError("No model directories found. Check --model-root / --model-glob.")

    rows: List[dict] = []
    masks: Dict[Tuple[str, int, str, int], np.ndarray] = {}

    for md in model_dirs:
        meta = _parse_model_meta(md)
        checkpoint_dirs = sorted([p for p in md.glob("checkpoint-*") if p.is_dir()])

        for ckpt_dir in checkpoint_dirs:
            ckpt = _checkpoint_num(ckpt_dir)
            if ckpt is None:
                continue

            pickles = sorted(ckpt_dir.glob(activation_glob))
            if not pickles:
                continue

            for pkl in pickles:
                try:
                    layer_map = _load_pickle(pkl)
                except Exception:
                    continue

                for layer, acts in sorted(layer_map.items()):
                    if selected_layers is not None and layer not in selected_layers:
                        continue
                    if acts.shape[0] < 3:
                        continue

                    layer_row, informative_mask = _compute_layer_metrics(acts)
                    activation_id = str(pkl.relative_to(ckpt_dir))
                    row = {
                        "model_dir": str(md),
                        "model_name": meta.model_name,
                        "training_type": meta.training_type,
                        "dataset": meta.dataset,
                        "checkpoint": ckpt,
                        "checkpoint_dir": str(ckpt_dir),
                        "activation_file": activation_id,
                        "layer": int(layer),
                        **layer_row,
                    }
                    rows.append(row)
                    masks[(str(md), ckpt, activation_id, int(layer))] = informative_mask

    if not rows:
        raise RuntimeError("No MP rows produced. Check activation pickles / globs.")

    df = pd.DataFrame(rows).sort_values(["model_name", "checkpoint", "activation_file", "layer"])

    transitions: List[dict] = []
    group_cols = ["model_dir", "activation_file", "layer"]
    for (model_dir_s, activation_id, layer), g in df.groupby(group_cols):
        g = g.sort_values("checkpoint")
        base_ckpt = int(g["checkpoint"].iloc[0])
        base_key = (model_dir_s, base_ckpt, activation_id, int(layer))
        base_mask = masks.get(base_key)
        if base_mask is None:
            continue

        base_signal_count = int(base_mask.sum())
        num_dims = int(base_mask.size)

        for _, row in g.iterrows():
            ckpt = int(row["checkpoint"])
            cur_key = (model_dir_s, ckpt, activation_id, int(layer))
            cur_mask = masks.get(cur_key)
            if cur_mask is None or cur_mask.size != base_mask.size:
                continue

            noise_to_signal = int((~base_mask & cur_mask).sum())
            signal_to_signal = int((base_mask & cur_mask).sum())
            signal_to_noise = int((base_mask & ~cur_mask).sum())
            noise_to_noise = int((~base_mask & ~cur_mask).sum())

            transitions.append(
                {
                    "model_dir": model_dir_s,
                    "model_name": row["model_name"],
                    "training_type": row["training_type"],
                    "dataset": row["dataset"],
                    "checkpoint": ckpt,
                    "baseline_checkpoint": base_ckpt,
                    "activation_file": activation_id,
                    "layer": int(layer),
                    "num_dims": num_dims,
                    "base_signal_count": base_signal_count,
                    "noise_to_signal": noise_to_signal,
                    "signal_to_signal": signal_to_signal,
                    "signal_to_noise": signal_to_noise,
                    "noise_to_noise": noise_to_noise,
                    "new_info_rate": float(noise_to_signal / num_dims) if num_dims else 0.0,
                    "reinforce_rate": float(signal_to_signal / base_signal_count)
                    if base_signal_count
                    else 0.0,
                }
            )

    tdf = pd.DataFrame(transitions)
    if tdf.empty:
        raise RuntimeError("Transition table is empty. Need >=1 checkpoint with matching masks.")

    summary = (
        df.groupby(["model_dir", "model_name", "training_type", "dataset", "checkpoint"], as_index=False)
        .agg(
            signal_fraction_eig_mean=("signal_fraction_eig", "mean"),
            signal_fraction_feature_mean=("signal_fraction_feature", "mean"),
            num_informative_feature_mean=("num_informative_feature", "mean"),
            lambda_max_mean=("lambda_max", "mean"),
            n_layers=("layer", "nunique"),
            n_activation_files=("activation_file", "nunique"),
        )
    )

    tsummary = (
        tdf.groupby(["model_dir", "model_name", "training_type", "dataset", "checkpoint"], as_index=False)
        .agg(
            new_info_rate_mean=("new_info_rate", "mean"),
            reinforce_rate_mean=("reinforce_rate", "mean"),
            noise_to_signal_total=("noise_to_signal", "sum"),
            signal_to_noise_total=("signal_to_noise", "sum"),
            num_dims_total=("num_dims", "sum"),
        )
    )

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    row_csv = out_dir / f"mp_checkpoint_rows_{tag}.csv"
    trans_csv = out_dir / f"mp_checkpoint_transitions_{tag}.csv"
    sum_csv = out_dir / f"mp_checkpoint_summary_{tag}.csv"
    merge_csv = out_dir / f"mp_checkpoint_summary_with_transitions_{tag}.csv"

    df.to_csv(row_csv, index=False)
    tdf.to_csv(trans_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    merged = summary.merge(
        tsummary,
        on=["model_dir", "model_name", "training_type", "dataset", "checkpoint"],
        how="left",
    ).sort_values(["model_name", "checkpoint"])
    merged.to_csv(merge_csv, index=False)

    print(f"Saved row-level MP metrics: {row_csv}")
    print(f"Saved transition metrics:   {trans_csv}")
    print(f"Saved summary metrics:      {sum_csv}")
    print(f"Saved merged summary:       {merge_csv}")


if __name__ == "__main__":
    app()
