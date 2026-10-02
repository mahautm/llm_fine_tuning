#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from paramem.probing.dii import compute_dii_matrix, load_layer_activations
from paramem.probing.pooled_split import build_pooled_vs_split_summary, load_partition_metrics

app = typer.Typer(add_completion=False)


def _plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="mako", cbar_kws={"label": "II"})
    plt.title(title)
    plt.xlabel("Right dataset")
    plt.ylabel("Left dataset")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


@app.command()
def main(
    activations_root: str = typer.Option(
        "results/mp_reservoir/activations", help="Root directory containing per-model activation pickles."
    ),
    spectra_root: str = typer.Option(
        "results/mp_reservoir/spectra", help="Root directory containing MP partition CSVs."
    ),
    output_root: str = typer.Option(
        "results/mp_reservoir/analysis", help="Output directory for pooled/split and DII analysis files."
    ),
    tag_suffix: str = typer.Option(
        "todo_exp1", help="Suffix used in mp_partition_metrics_<model>_<suffix>.csv names."
    ),
    k_neighbors: int = typer.Option(10, help="k value for information imbalance computation."),
    max_samples: int = typer.Option(2000, help="Maximum samples per dataset/layer used for DII embeddings."),
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    activations_root_path = Path(activations_root)
    spectra_root_path = Path(spectra_root)
    output_root_path = Path(output_root)

    model_dirs = sorted([p for p in activations_root_path.iterdir() if p.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"No model directories found under {activations_root_path}")

    for model_dir in model_dirs:
        model_tag = model_dir.name
        metrics_path = spectra_root_path / model_tag / f"mp_partition_metrics_{model_tag}_{tag_suffix}.csv"
        if not metrics_path.exists():
            continue

        partition_metrics = load_partition_metrics(metrics_path)
        dataset_to_pickle = {
            p.stem.split("_layers_")[0]: p
            for p in sorted(model_dir.glob("*.pickle"))
            if "layers_" in p.stem
        }
        if not dataset_to_pickle:
            continue

        model_output = output_root_path / model_tag
        model_output.mkdir(parents=True, exist_ok=True)

        pooled_split_df = build_pooled_vs_split_summary(dataset_to_pickle, partition_metrics)
        pooled_csv = model_output / f"pooled_vs_split_{model_tag}_{tag_suffix}.csv"
        pooled_split_df.to_csv(pooled_csv, index=False)

        pooled_plot = model_output / f"pooled_vs_split_{model_tag}_{tag_suffix}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(pooled_split_df["layer"], pooled_split_df["pooled_signal_fraction"], marker="o", label="Pooled")
        plt.plot(pooled_split_df["layer"], pooled_split_df["split_mean_signal_fraction"], marker="o", label="Split mean")
        plt.fill_between(
            pooled_split_df["layer"],
            pooled_split_df["split_mean_signal_fraction"] - pooled_split_df["split_signal_fraction_std"],
            pooled_split_df["split_mean_signal_fraction"] + pooled_split_df["split_signal_fraction_std"],
            alpha=0.2,
            label="Split std",
        )
        plt.xlabel("Layer")
        plt.ylabel("Signal Fraction")
        plt.title(f"Pooled vs Split MP Signal Fraction ({model_tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pooled_plot, dpi=220)
        plt.close()

        mean_matrices, directional_tables = compute_dii_matrix(
            dataset_to_pickle,
            partition_metrics,
            k_neighbors=k_neighbors,
            max_samples=max_samples,
        )

        for layer, matrix in mean_matrices.items():
            matrix_csv = model_output / f"dii_matrix_layer_{layer}_{tag_suffix}.csv"
            matrix.to_csv(matrix_csv)
            matrix_png = model_output / f"dii_matrix_layer_{layer}_{tag_suffix}.png"
            _plot_heatmap(matrix, f"DII Matrix - {model_tag} - layer {layer}", matrix_png)

            dir_csv = model_output / f"dii_pairs_layer_{layer}_{tag_suffix}.csv"
            directional_tables[layer].to_csv(dir_csv, index=False)

        print(f"Saved pooled/split analysis and DII matrices for {model_tag} to {model_output}")


if __name__ == "__main__":
    app()
