#!/usr/bin/env python3
"""Cross-model FT-vs-LoRA comparison for MP and memorization metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PAIRS = [
    {
        "pair": "pythia70m_wikidata",
        "ft_root": Path("/home/mmahaut/projects/paramem/models_smoke/pythia70m-fsdp-0-wikidata"),
        "lora_root": Path("/home/mmahaut/projects/paramem/models_smoke/pythia70m-fsdp-1-wikidata"),
    },
    {
        "pair": "llama8b_wikiplus",
        "ft_root": Path("/home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-0-wikiplus-lr1e-4-rerun4"),
        "lora_root": Path("/home/mmahaut/projects/paramem/models4/Llama-3.1-8B-Instruct-fsdp-1-wikiplus-lr1e-4"),
    },
]

METRICS = [
    "signal_fraction_feature",
    "num_informative_feature",
    "train_nll",
    "exposure_delta",
    "avg_tokens_recovered",
]

METRIC_LABELS = {
    "signal_fraction_feature": "Signal Fraction (Feature Basis)",
    "num_informative_feature": "# Informative Features",
    "train_nll": "Train NLL",
    "exposure_delta": "Exposure Delta",
    "avg_tokens_recovered": "Avg Tokens Recovered",
}


def _checkpoint_num(path: Path) -> int:
    return int(path.name.split("-")[1])


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_layer_metric(metrics: List[dict], preferred_layer: int = 3) -> dict:
    for m in metrics:
        if int(m.get("layer", -1)) == preferred_layer:
            return m
    return metrics[0] if metrics else {}


def collect_table() -> pd.DataFrame:
    rows = []
    for pair_cfg in PAIRS:
        pair = pair_cfg["pair"]
        for mode, root in [("FT", pair_cfg["ft_root"]), ("LoRA", pair_cfg["lora_root"])]:
            if not root.exists():
                continue
            root_state = _read_json(root / "trainer_state.json")
            root_max_steps = int(root_state.get("max_steps", 0))
            root_num_epochs = float(root_state.get("num_train_epochs", 1.0))
            root_train_batch = int(root_state.get("train_batch_size", 4 if mode == "FT" else 2))
            ckpts = sorted(root.glob("checkpoint-*"), key=_checkpoint_num)
            for ckpt_dir in ckpts:
                ck = _checkpoint_num(ckpt_dir)
                slurm_dir = ckpt_dir / "slurm_logs"
                mem_p = slurm_dir / "memorization_metrics.json"
                mp_p = slurm_dir / "mp_checkpoint_metrics.json"
                state_p = ckpt_dir / "trainer_state.json"
                if not (mem_p.exists() and mp_p.exists()):
                    continue

                mem = _read_json(mem_p)
                mp = _read_json(mp_p)
                if not isinstance(mem, dict) or mem == {}:
                    continue

                layer_metrics = _pick_layer_metric(mp.get("metrics", []), preferred_layer=3)
                if not layer_metrics:
                    continue

                state = _read_json(state_p) if state_p.exists() else {}
                global_step = int(state.get("global_step", ck))
                max_steps = int(state.get("max_steps", root_max_steps if root_max_steps > 0 else ck))
                if max_steps <= 0:
                    max_steps = ck
                if "epoch" in state:
                    dataset_passes = float(state["epoch"])
                elif root_max_steps > 0:
                    dataset_passes = float((global_step / root_max_steps) * root_num_epochs)
                else:
                    dataset_passes = float(global_step / max_steps if max_steps > 0 else 0.0)

                train_batch_size = int(state.get("train_batch_size", root_train_batch))

                rows.append(
                    {
                        "pair": pair,
                        "mode": mode,
                        "checkpoint": ck,
                        "global_step": global_step,
                        "max_steps": max_steps,
                        "dataset_passes_seen": dataset_passes,
                        "train_batch_size": train_batch_size,
                        "exposure_units": ck * train_batch_size,
                        "signal_fraction_feature": float(layer_metrics.get("signal_fraction_feature", np.nan)),
                        "num_informative_feature": float(layer_metrics.get("num_informative_feature", np.nan)),
                        "lambda_max": float(layer_metrics.get("lambda_max", np.nan)),
                        "extractability_rate": float(mem.get("extractability_rate", np.nan)),
                        "avg_tokens_recovered": float(mem.get("avg_tokens_recovered", np.nan)),
                        "ngram_correlation": float(mem.get("ngram_correlation", np.nan)),
                        "train_nll": float(mem.get("train_nll", np.nan)),
                        "exposure_delta": float(mem.get("exposure_delta", np.nan)),
                    }
                )

    return pd.DataFrame(rows).sort_values(["pair", "mode", "checkpoint"]).reset_index(drop=True)


def exact_matched_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, g in df.groupby("pair"):
        ft = g[g["mode"] == "FT"].copy()
        lo = g[g["mode"] == "LoRA"].copy()
        if ft.empty or lo.empty:
            continue
        ft["pass_key"] = ft["dataset_passes_seen"].round(6)
        lo["pass_key"] = lo["dataset_passes_seen"].round(6)
        common = sorted(set(ft["pass_key"]) & set(lo["pass_key"]))
        for pk in common:
            rf = ft[ft["pass_key"] == pk].iloc[0]
            rl = lo[lo["pass_key"] == pk].iloc[0]
            row = {
                "pair": pair,
                "dataset_passes_seen": float(pk),
            }
            for metric in METRICS:
                row[f"delta_{metric}_FT_minus_LoRA"] = float(rf[metric] - rl[metric])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["pair", "dataset_passes_seen"]).reset_index(drop=True)


def interpolated_deltas(df: pd.DataFrame, n_points: int = 25) -> pd.DataFrame:
    rows = []
    for pair, g in df.groupby("pair"):
        ft = g[g["mode"] == "FT"].sort_values("dataset_passes_seen")
        lo = g[g["mode"] == "LoRA"].sort_values("dataset_passes_seen")
        if len(ft) < 2 or len(lo) < 2:
            continue
        start = max(float(ft["dataset_passes_seen"].min()), float(lo["dataset_passes_seen"].min()))
        end = min(float(ft["dataset_passes_seen"].max()), float(lo["dataset_passes_seen"].max()))
        if end <= start:
            continue

        grid = np.linspace(start, end, n_points)
        row_base = {
            "pair": pair,
        }
        for x in grid:
            row = dict(row_base)
            row["dataset_passes_seen"] = float(x)
            for metric in METRICS:
                ft_y = np.interp(x, ft["dataset_passes_seen"].to_numpy(), ft[metric].to_numpy())
                lo_y = np.interp(x, lo["dataset_passes_seen"].to_numpy(), lo[metric].to_numpy())
                row[f"delta_{metric}_FT_minus_LoRA"] = float(ft_y - lo_y)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["pair", "dataset_passes_seen"]).reset_index(drop=True)


def plot_pair_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    metrics = [
        "signal_fraction_feature",
        "num_informative_feature",
        "train_nll",
        "exposure_delta",
        "avg_tokens_recovered",
    ]
    pairs = sorted(df["pair"].unique())
    fig, axes = plt.subplots(len(pairs), len(metrics), figsize=(4 * len(metrics), 3.4 * len(pairs)), dpi=150)
    if len(pairs) == 1:
        axes = np.expand_dims(axes, 0)

    for i, pair in enumerate(pairs):
        sub = df[df["pair"] == pair]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            sns.lineplot(
                data=sub,
                x="dataset_passes_seen",
                y=metric,
                hue="mode",
                marker="o",
                linewidth=2,
                ax=ax,
            )
            if i == 0:
                ax.set_title(METRIC_LABELS[metric])
            if j == 0:
                ax.set_ylabel(pair)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Dataset Passes")
            if i != 0 or j != 0:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def plot_interpolated_deltas(df_interp: pd.DataFrame, out_path: Path) -> None:
    if df_interp.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flatten()
    for ax, metric in zip(axes, METRICS):
        col = f"delta_{metric}_FT_minus_LoRA"
        sns.lineplot(
            data=df_interp,
            x="dataset_passes_seen",
            y=col,
            hue="pair",
            linewidth=2,
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Delta FT-LoRA: {METRIC_LABELS[metric]}")
        ax.set_xlabel("Dataset Passes")
        ax.set_ylabel("Delta")
    if len(METRICS) < len(axes):
        for ax in axes[len(METRICS) :]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    out_dir = Path("/home/mmahaut/projects/paramem/results/mp_reservoir/ft_lora_dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    table = collect_table()
    table_csv = out_dir / "cross_model_ft_lora_mp_mem_table.csv"
    table.to_csv(table_csv, index=False)

    exact = exact_matched_deltas(table)
    exact_csv = out_dir / "cross_model_matched_pass_deltas.csv"
    exact.to_csv(exact_csv, index=False)

    interp = interpolated_deltas(table, n_points=25)
    interp_csv = out_dir / "cross_model_interpolated_pass_deltas.csv"
    interp.to_csv(interp_csv, index=False)

    auc_rows = []
    for pair, g in interp.groupby("pair"):
        x = g["dataset_passes_seen"].to_numpy(dtype=float)
        for metric in METRICS:
            col = f"delta_{metric}_FT_minus_LoRA"
            auc_rows.append(
                {
                    "pair": pair,
                    "metric": metric,
                    "auc_delta_ft_minus_lora": float(np.trapz(g[col].to_numpy(dtype=float), x)),
                }
            )
    auc_df = pd.DataFrame(auc_rows)
    auc_csv = out_dir / "cross_model_auc_deltas.csv"
    auc_df.to_csv(auc_csv, index=False)

    plot_pair_trajectories(table, plot_dir / "cross_model_pair_trajectories.png")
    plot_interpolated_deltas(interp, plot_dir / "cross_model_interpolated_deltas.png")

    print(f"Saved: {table_csv}")
    print(f"Saved: {exact_csv}")
    print(f"Saved: {interp_csv}")
    print(f"Saved: {auc_csv}")
    print(f"Saved: {plot_dir / 'cross_model_pair_trajectories.png'}")
    print(f"Saved: {plot_dir / 'cross_model_interpolated_deltas.png'}")
    print("Rows:")
    print(table.groupby(["pair", "mode"]).size().to_string())


if __name__ == "__main__":
    main()
