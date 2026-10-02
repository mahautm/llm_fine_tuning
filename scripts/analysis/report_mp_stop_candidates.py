#!/usr/bin/env python3
"""
Report MP-informed early stop candidates from a merged checkpoint table.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _pick_perf_column(df: pd.DataFrame, preferred: str) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if c.endswith("_accuracy")]
    if candidates:
        if "mmlu_accuracy" in candidates:
            return "mmlu_accuracy"
        return candidates[0]
    return None


def _stop_checkpoint_for_group(
    g: pd.DataFrame,
    perf_col: Optional[str],
    overfit_col: Optional[str],
    k_consecutive: int,
    epsilon_mp: float,
    epsilon_perf: float,
) -> tuple[Optional[int], str]:
    g = g.sort_values("checkpoint").copy()
    g["delta_mp"] = g["signal_fraction_feature_mean"].diff().abs()

    if perf_col is not None:
        g["delta_perf"] = g[perf_col].diff()
        perf_small = g["delta_perf"] <= epsilon_perf
    else:
        perf_small = pd.Series([True] * len(g), index=g.index)

    if overfit_col is not None and overfit_col in g.columns:
        overfit_up = g[overfit_col].diff() > 0
    else:
        overfit_up = pd.Series([True] * len(g), index=g.index)

    mp_small = g["delta_mp"] <= epsilon_mp

    cond = mp_small & perf_small & overfit_up
    cond = cond.fillna(False)

    # Find first checkpoint where condition holds for k consecutive checkpoints.
    run = 0
    for idx, ok in zip(g.index, cond):
        run = run + 1 if ok else 0
        if run >= k_consecutive:
            ckpt = int(g.loc[idx, "checkpoint"])
            return ckpt, "mp_plateau+low_perf_gain+overfit_risk"

    return None, "no_candidate"


@app.command()
def main(
    merged_csv: str = typer.Option(..., help="Merged table from merge_mp_mem_perf_checkpoint_table.py"),
    output_csv: str = typer.Option(
        "results/mp_reservoir/ft_lora_dynamics/mp_stop_candidates.csv",
        help="Output CSV for stop candidates.",
    ),
    perf_metric: str = typer.Option(
        "mmlu_accuracy",
        help="Performance metric column to use for marginal gain criterion. Falls back to first *_accuracy.",
    ),
    overfit_metric: str = typer.Option(
        "exposure_delta",
        help="Overfit pressure column (higher means more memorization pressure).",
    ),
    k_consecutive: int = typer.Option(2, help="Require condition for K consecutive checkpoints."),
    epsilon_mp: float = typer.Option(
        0.002,
        help="Threshold for |delta signal_fraction_feature_mean| to consider MP dynamics plateaued.",
    ),
    epsilon_perf: float = typer.Option(
        0.002,
        help="Threshold for marginal performance gain to consider improvement negligible.",
    ),
) -> None:
    df = pd.read_csv(merged_csv)

    required = {"model_dir", "model_name", "training_type", "dataset", "checkpoint", "signal_fraction_feature_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    perf_col = _pick_perf_column(df, perf_metric)
    overfit_col = overfit_metric if overfit_metric in df.columns else None

    rows: List[dict] = []

    for (model_dir, model_name, training_type, dataset), g in df.groupby(
        ["model_dir", "model_name", "training_type", "dataset"],
        as_index=False,
    ):
        stop_ckpt, reason = _stop_checkpoint_for_group(
            g,
            perf_col=perf_col,
            overfit_col=overfit_col,
            k_consecutive=k_consecutive,
            epsilon_mp=epsilon_mp,
            epsilon_perf=epsilon_perf,
        )

        last_ckpt = int(g["checkpoint"].max())
        rows.append(
            {
                "model_dir": model_dir,
                "model_name": model_name,
                "training_type": training_type,
                "dataset": dataset,
                "perf_metric_used": perf_col or "none",
                "overfit_metric_used": overfit_col or "none",
                "k_consecutive": k_consecutive,
                "epsilon_mp": epsilon_mp,
                "epsilon_perf": epsilon_perf,
                "stop_checkpoint": stop_ckpt if stop_ckpt is not None else np.nan,
                "last_checkpoint": last_ckpt,
                "reason": reason,
            }
        )

    out = pd.DataFrame(rows).sort_values(["training_type", "dataset", "model_name"])
    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved stop-candidate report: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    app()
