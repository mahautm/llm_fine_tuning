#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

app = typer.Typer(add_completion=False)


def _plot_single_heatmap(df: pd.DataFrame, model_tag: str, out_path: Path) -> None:
    pivot = df.pivot(index="layer", columns="dataset", values="signal_fraction")
    pivot = pivot.sort_index()

    plt.figure(figsize=(9, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Signal Fraction"},
    )
    plt.title(f"MP Exp1 Signal Fraction Heatmap ({model_tag})")
    plt.xlabel("Dataset")
    plt.ylabel("Layer")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_combined_summary(dfs: List[pd.DataFrame], tags: List[str], out_path: Path) -> None:
    merged = []
    for tag, df in zip(tags, dfs):
        local = df.copy()
        local["model_tag"] = tag
        merged.append(local)

    full = pd.concat(merged, ignore_index=True)

    g = sns.FacetGrid(full, col="model_tag", sharex=False, sharey=True, height=4, aspect=1.4)
    g.map_dataframe(sns.lineplot, x="layer", y="signal_fraction", hue="dataset", marker="o")
    g.add_legend()
    g.set_axis_labels("Layer", "Signal Fraction")
    g.set_titles("{col_name}")
    g.figure.suptitle("MP Exp1 Signal Fraction by Layer and Dataset", y=1.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(g.figure)


@app.command()
def main(
    spectra_root: str = typer.Option(
        "results/mp_reservoir/spectra", help="Root folder containing per-model MP metrics CSVs."
    ),
    figures_root: str = typer.Option(
        "results/mp_reservoir/figures", help="Output folder for generated figures."
    ),
    tag_suffix: str = typer.Option(
        "todo_exp1", help="Suffix used in mp_partition_metrics_<model>_<suffix>.csv names."
    ),
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    spectra_root_path = Path(spectra_root)
    figures_root_path = Path(figures_root)

    model_dirs = sorted([p for p in spectra_root_path.iterdir() if p.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"No model directories found under {spectra_root_path}")

    all_dfs: List[pd.DataFrame] = []
    all_tags: List[str] = []

    for model_dir in model_dirs:
        model_tag = model_dir.name
        csv_path = model_dir / f"mp_partition_metrics_{model_tag}_{tag_suffix}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        out_file = figures_root_path / model_tag / f"mp_signal_fraction_heatmap_{model_tag}_{tag_suffix}.png"
        _plot_single_heatmap(df, model_tag, out_file)

        all_dfs.append(df)
        all_tags.append(model_tag)

    if not all_dfs:
        raise RuntimeError("No MP metrics CSV files found for plotting.")

    summary_out = figures_root_path / f"mp_signal_fraction_layer_trends_{tag_suffix}.png"
    _plot_combined_summary(all_dfs, all_tags, summary_out)

    print(f"Generated {len(all_dfs)} heatmaps + summary plot under {figures_root_path}")


if __name__ == "__main__":
    app()
