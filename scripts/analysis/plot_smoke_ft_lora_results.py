#!/usr/bin/env python3
"""Generate compact analysis plots for FT vs LoRA smoke MP results."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_root_for_mode(mode: str) -> Path:
    base = Path("/home/mmahaut/projects/paramem/models_smoke")
    if mode == "FT":
        return base / "pythia70m-fsdp-0-wikidata"
    if mode == "LoRA":
        return base / "pythia70m-fsdp-1-wikidata"
    raise ValueError(f"Unknown mode '{mode}'")


def _load_train_batch_size(mode: str, checkpoint: int) -> int:
    root = _run_root_for_mode(mode)
    state_path = root / f"checkpoint-{int(checkpoint)}" / "trainer_state.json"
    if not state_path.exists():
        # Fallback defaults based on observed smoke configs.
        return 4 if mode == "FT" else 2
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    val = state.get("train_batch_size", 4 if mode == "FT" else 2)
    try:
        return int(val)
    except Exception:
        return 4 if mode == "FT" else 2


def _load_trainer_state(mode: str, checkpoint: int) -> dict:
    root = _run_root_for_mode(mode)
    state_path = root / f"checkpoint-{int(checkpoint)}" / "trainer_state.json"
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    in_csv = Path("results/mp_reservoir/ft_lora_dynamics/smoke_subset_mp_mem_summary.csv")
    out_dir = Path("results/mp_reservoir/ft_lora_dynamics/plots")
    _ensure_dir(out_dir)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise RuntimeError(f"Input CSV is empty: {in_csv}")

    df = df.sort_values(["mode", "checkpoint"]).copy()

    # Trainer-derived progress metrics.
    trainer_states = [_load_trainer_state(m, c) for m, c in zip(df["mode"], df["checkpoint"])]
    df["train_batch_size"] = [
        int(s.get("train_batch_size", _load_train_batch_size(m, c)))
        for s, m, c in zip(trainer_states, df["mode"], df["checkpoint"])
    ]
    df["global_step"] = [int(s.get("global_step", int(c))) for s, c in zip(trainer_states, df["checkpoint"])]
    df["max_steps"] = [int(s.get("max_steps", int(c))) if int(s.get("max_steps", int(c))) > 0 else int(c) for s, c in zip(trainer_states, df["checkpoint"])]
    df["dataset_passes_seen"] = [
        float(s.get("epoch", gs / ms if ms > 0 else 0.0))
        for s, gs, ms in zip(trainer_states, df["global_step"], df["max_steps"])
    ]
    df["exposure_units"] = df["checkpoint"] * df["train_batch_size"]

    # Normalize x-axis by in-run data exposure proxy.
    # checkpoint is proportional to optimizer update count given fixed save cadence.
    mode_max = df.groupby("mode")["checkpoint"].transform("max")
    mode_min = df.groupby("mode")["checkpoint"].transform("min")
    denom = (mode_max - mode_min).replace(0, 1)
    df["exposure_frac"] = (df["checkpoint"] - mode_min) / denom
    df["exposure_pct"] = 100.0 * df["exposure_frac"]

    # Also normalize by trainer-derived exposure units for cross-mode comparability.
    mode_max_expo = df.groupby("mode")["exposure_units"].transform("max")
    mode_min_expo = df.groupby("mode")["exposure_units"].transform("min")
    expo_denom = (mode_max_expo - mode_min_expo).replace(0, 1)
    df["exposure_units_frac"] = (df["exposure_units"] - mode_min_expo) / expo_denom
    df["exposure_units_pct"] = 100.0 * df["exposure_units_frac"]

    # Normalize by dataset passes (epochs seen), the clearest cross-regime unit.
    mode_max_passes = df.groupby("mode")["dataset_passes_seen"].transform("max")
    mode_min_passes = df.groupby("mode")["dataset_passes_seen"].transform("min")
    pass_denom = (mode_max_passes - mode_min_passes).replace(0, 1)
    df["dataset_passes_frac"] = (df["dataset_passes_seen"] - mode_min_passes) / pass_denom
    df["dataset_passes_pct"] = 100.0 * df["dataset_passes_frac"]

    # Keep a long-format table for trajectory plots.
    metric_labels = {
        "signal_fraction_feature": "Signal Fraction (Feature Basis)",
        "num_informative_feature": "# Informative Features",
        "train_nll": "Train NLL",
        "exposure_delta": "Exposure Delta",
        "avg_tokens_recovered": "Avg Tokens Recovered",
        "ngram_correlation": "N-gram Correlation",
    }

    sns.set_theme(style="whitegrid")

    # 1) MP trajectory: signal fraction + informative count
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for ax, metric in zip(axes, ["signal_fraction_feature", "num_informative_feature"]):
        sns.lineplot(
            data=df,
            x="checkpoint",
            y=metric,
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(metric_labels[metric])
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(metric_labels[metric])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mp_trajectory.png")
    plt.close(fig)

    # 1b) MP trajectory on normalized exposure axis (recommended for FT vs LoRA comparability)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for ax, metric in zip(axes, ["signal_fraction_feature", "num_informative_feature"]):
        sns.lineplot(
            data=df,
            x="exposure_pct",
            y=metric,
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"{metric_labels[metric]} (Normalized)")
        ax.set_xlabel("Data Exposure (% of run)")
        ax.set_ylabel(metric_labels[metric])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mp_trajectory_normalized.png")
    plt.close(fig)

    # 1c) MP trajectory on dataset-passes axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for ax, metric in zip(axes, ["signal_fraction_feature", "num_informative_feature"]):
        sns.lineplot(
            data=df,
            x="dataset_passes_seen",
            y=metric,
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"{metric_labels[metric]} (Dataset Passes)")
        ax.set_xlabel("Dataset Passes Seen (epochs)")
        ax.set_ylabel(metric_labels[metric])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mp_trajectory_dataset_passes.png")
    plt.close(fig)

    # 2) Memorization trajectory
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, metric in zip(axes, ["train_nll", "exposure_delta", "avg_tokens_recovered"]):
        sns.lineplot(
            data=df,
            x="checkpoint",
            y=metric,
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(metric_labels[metric])
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(metric_labels[metric])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mem_trajectory.png")
    plt.close(fig)

    # 2b) Memorization trajectory on normalized exposure axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, metric in zip(axes, ["train_nll", "exposure_delta", "avg_tokens_recovered"]):
        sns.lineplot(
            data=df,
            x="exposure_pct",
            y=metric,
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"{metric_labels[metric]} (Normalized)")
        ax.set_xlabel("Data Exposure (% of run)")
        ax.set_ylabel(metric_labels[metric])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mem_trajectory_normalized.png")
    plt.close(fig)

    # 2c) Memorization trajectory on dataset-passes axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, metric in zip(axes, ["train_nll", "exposure_delta", "avg_tokens_recovered"]):
        sns.lineplot(
            data=df,
            x="dataset_passes_seen",
            y=metric,
            hue="mode",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"{metric_labels[metric]} (Dataset Passes)")
        ax.set_xlabel("Dataset Passes Seen (epochs)")
        ax.set_ylabel(metric_labels[metric])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mem_trajectory_dataset_passes.png")
    plt.close(fig)

    # 2d) Memorization trajectories on shared matched dataset-passes window.
    # Compare only up to the minimum max dataset passes across modes.
    max_common_passes = df.groupby("mode")["dataset_passes_seen"].max().min()
    matched = df[df["dataset_passes_seen"] <= max_common_passes].copy()
    if not matched.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
        for ax, metric in zip(axes, ["train_nll", "exposure_delta", "avg_tokens_recovered"]):
            sns.lineplot(
                data=matched,
                x="dataset_passes_seen",
                y=metric,
                hue="mode",
                marker="o",
                linewidth=2,
                ax=ax,
            )
            ax.set_title(f"{metric_labels[metric]} (Matched Dataset Passes)")
            ax.set_xlabel("Dataset Passes Seen (epochs)")
            ax.set_ylabel(metric_labels[metric])
        fig.tight_layout()
        fig.savefig(out_dir / "smoke_mem_trajectory_matched_dataset_passes.png")
        plt.close(fig)
        matched.to_csv(out_dir / "smoke_subset_matched_dataset_passes.csv", index=False)

    # 3) Coupling plot: MP informative fraction vs memorization pressure
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    sns.scatterplot(
        data=df,
        x="signal_fraction_feature",
        y="exposure_delta",
        hue="mode",
        style="mode",
        s=90,
        ax=ax,
    )
    for _, r in df.iterrows():
        ax.annotate(f"{r['mode']}-{int(r['checkpoint'])}", (r["signal_fraction_feature"], r["exposure_delta"]), fontsize=7, alpha=0.75)
    ax.set_title("Coupling: MP Signal Fraction vs Exposure Delta")
    ax.set_xlabel(metric_labels["signal_fraction_feature"])
    ax.set_ylabel(metric_labels["exposure_delta"])
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_mp_vs_exposure_scatter.png")
    plt.close(fig)

    # 3b) Phase portraits over dataset passes.
    # This directly shows MP-to-memorization coupling trajectories by regime.
    phase_specs = [
        ("num_informative_feature", "exposure_delta", "MP Features -> Exposure Delta"),
        ("signal_fraction_feature", "avg_tokens_recovered", "MP Signal Fraction -> Tokens Recovered"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for ax, (x_metric, y_metric, title) in zip(axes, phase_specs):
        for mode, g in df.groupby("mode"):
            g = g.sort_values("dataset_passes_seen")
            ax.plot(
                g[x_metric],
                g[y_metric],
                marker="o",
                linewidth=2,
                label=mode,
            )
            for _, r in g.iterrows():
                ax.annotate(
                    f"{r['dataset_passes_seen']:.2f}",
                    (r[x_metric], r[y_metric]),
                    fontsize=7,
                    alpha=0.7,
                )
        ax.set_title(title)
        ax.set_xlabel(metric_labels.get(x_metric, x_metric))
        ax.set_ylabel(metric_labels.get(y_metric, y_metric))
    axes[0].legend(title="Mode")
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_phase_portraits_dataset_passes.png")
    plt.close(fig)

    # 4) Consecutive deltas by mode
    delta_rows = []
    for mode, g in df.groupby("mode"):
        g = g.sort_values("checkpoint")
        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            cur = g.iloc[i]
            delta_rows.append(
                {
                    "mode": mode,
                    "from_ckpt": int(prev["checkpoint"]),
                    "to_ckpt": int(cur["checkpoint"]),
                    "delta_signal_fraction_feature": float(cur["signal_fraction_feature"] - prev["signal_fraction_feature"]),
                    "delta_num_informative_feature": float(cur["num_informative_feature"] - prev["num_informative_feature"]),
                    "delta_exposure_delta": float(cur["exposure_delta"] - prev["exposure_delta"]),
                    "delta_train_nll": float(cur["train_nll"] - prev["train_nll"]),
                }
            )

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
        delta_df["step"] = delta_df["from_ckpt"].astype(str) + "->" + delta_df["to_ckpt"].astype(str)

        sns.barplot(data=delta_df, x="step", y="delta_signal_fraction_feature", hue="mode", ax=axes[0])
        axes[0].set_title("Delta Signal Fraction per Step")
        axes[0].set_xlabel("Checkpoint Step")
        axes[0].set_ylabel("Delta")

        sns.barplot(data=delta_df, x="step", y="delta_num_informative_feature", hue="mode", ax=axes[1])
        axes[1].set_title("Delta # Informative Features per Step")
        axes[1].set_xlabel("Checkpoint Step")
        axes[1].set_ylabel("Delta")

        fig.tight_layout()
        fig.savefig(out_dir / "smoke_deltas.png")
        plt.close(fig)

        delta_df.to_csv(out_dir / "smoke_deltas.csv", index=False)

    # 5) Summary stats table
    summary = (
        df.groupby("mode", as_index=False)[
            [
                "signal_fraction_feature",
                "num_informative_feature",
                "train_nll",
                "exposure_delta",
                "avg_tokens_recovered",
                "ngram_correlation",
            ]
        ]
        .agg(["mean", "min", "max"])
    )
    summary.to_csv(out_dir / "smoke_summary_stats.csv")

    # 6) Matched-pass deltas and compact summary figures.
    compare_metrics = [
        "signal_fraction_feature",
        "num_informative_feature",
        "train_nll",
        "exposure_delta",
        "avg_tokens_recovered",
    ]
    ft = df[df["mode"] == "FT"][["dataset_passes_seen", *compare_metrics]].copy()
    lo = df[df["mode"] == "LoRA"][["dataset_passes_seen", *compare_metrics]].copy()
    ft["pass_key"] = ft["dataset_passes_seen"].round(6)
    lo["pass_key"] = lo["dataset_passes_seen"].round(6)
    common_passes = sorted(set(ft["pass_key"]) & set(lo["pass_key"]))

    if common_passes:
        ft_m = ft[ft["pass_key"].isin(common_passes)].sort_values("pass_key")
        lo_m = lo[lo["pass_key"].isin(common_passes)].sort_values("pass_key")
        cmp_df = ft_m.merge(lo_m, on="pass_key", suffixes=("_FT", "_LoRA"))
        cmp_df["dataset_passes_seen"] = cmp_df["dataset_passes_seen_FT"]
        for metric in compare_metrics:
            cmp_df[f"delta_{metric}_FT_minus_LoRA"] = (
                cmp_df[f"{metric}_FT"] - cmp_df[f"{metric}_LoRA"]
            )

        cmp_df.to_csv(out_dir / "smoke_matched_pass_comparison.csv", index=False)

        delta_long = []
        for metric in compare_metrics:
            for _, r in cmp_df.iterrows():
                delta_long.append(
                    {
                        "dataset_passes_seen": float(r["dataset_passes_seen"]),
                        "metric": metric,
                        "delta_ft_minus_lora": float(r[f"delta_{metric}_FT_minus_LoRA"]),
                    }
                )
        delta_long_df = pd.DataFrame(delta_long)
        delta_long_df.to_csv(out_dir / "smoke_matched_pass_deltas.csv", index=False)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
        axes_list = list(axes.flatten())
        for ax, metric in zip(axes_list, compare_metrics):
            sub = delta_long_df[delta_long_df["metric"] == metric]
            sns.barplot(
                data=sub,
                x="dataset_passes_seen",
                y="delta_ft_minus_lora",
                color="#457b9d",
                ax=ax,
            )
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_title(f"FT-LoRA: {metric_labels.get(metric, metric)}")
            ax.set_xlabel("Dataset Passes")
            ax.set_ylabel("Delta")
        for ax in axes_list[len(compare_metrics) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "smoke_matched_pass_deltas.png")
        plt.close(fig)

        # AUC over matched passes for compact single-number comparison.
        auc_rows = []
        x = cmp_df["dataset_passes_seen"].to_numpy(dtype=float)
        for metric in compare_metrics:
            y_ft = cmp_df[f"{metric}_FT"].to_numpy(dtype=float)
            y_lo = cmp_df[f"{metric}_LoRA"].to_numpy(dtype=float)
            auc_ft = float(np.trapz(y_ft, x))
            auc_lo = float(np.trapz(y_lo, x))
            auc_rows.append(
                {
                    "metric": metric,
                    "auc_ft": auc_ft,
                    "auc_lora": auc_lo,
                    "auc_delta_ft_minus_lora": auc_ft - auc_lo,
                }
            )
        auc_df = pd.DataFrame(auc_rows)
        auc_df.to_csv(out_dir / "smoke_auc_matched_passes.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
        sns.barplot(
            data=auc_df,
            x="metric",
            y="auc_delta_ft_minus_lora",
            color="#1d3557",
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("AUC Delta on Matched Dataset Passes (FT - LoRA)")
        ax.set_xlabel("Metric")
        ax.set_ylabel("AUC Delta")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / "smoke_auc_delta_bars.png")
        plt.close(fig)

    # Save normalized plotting table for downstream/manual plotting.
    df.to_csv(out_dir / "smoke_subset_with_normalized_exposure.csv", index=False)
    df.to_csv(out_dir / "smoke_subset_with_dataset_passes.csv", index=False)

    print(f"Saved plots and tables to: {out_dir}")


if __name__ == "__main__":
    main()
