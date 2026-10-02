#!/usr/bin/env python3
"""
Merge checkpoint-level MP dynamics, memorization metrics, and performance metrics
into a unified table for FT-vs-LoRA analysis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _checkpoint_num(path: Path) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)", str(path))
    return int(m.group(1)) if m else None


def _parse_model_meta(model_dir: Path) -> Dict[str, str]:
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

    return {
        "model_name": name,
        "training_type": training_type,
        "dataset": dataset,
    }


def _collect_mem_rows(model_roots: List[Path]) -> pd.DataFrame:
    rows = []
    for root in model_roots:
        if not root.exists():
            continue
        for model_dir in root.glob("*fsdp-*"):
            if not model_dir.is_dir():
                continue
            meta = _parse_model_meta(model_dir)
            for ckpt_dir in model_dir.glob("checkpoint-*"):
                ckpt = _checkpoint_num(ckpt_dir)
                if ckpt is None:
                    continue
                mem_json = ckpt_dir / "slurm_logs" / "memorization_metrics.json"
                if not mem_json.exists():
                    continue
                try:
                    payload = json.loads(mem_json.read_text(encoding="utf-8"))
                except Exception:
                    continue
                rows.append(
                    {
                        "model_dir": str(model_dir),
                        "checkpoint": ckpt,
                        "model_name": meta["model_name"],
                        "training_type": meta["training_type"],
                        "dataset": meta["dataset"],
                        "extractability_rate": payload.get("extractability_rate", np.nan),
                        "avg_tokens_recovered": payload.get("avg_tokens_recovered", np.nan),
                        "ngram_correlation": payload.get("ngram_correlation", np.nan),
                        "intrinsic_dimension": payload.get("intrinsic_dimension", np.nan),
                        "train_nll": payload.get("train_nll", np.nan),
                        "train_perplexity": payload.get("train_perplexity", np.nan),
                        "exposure_delta": payload.get("exposure_delta", np.nan),
                        "exposure_positive_rate": payload.get("exposure_positive_rate", np.nan),
                        "num_sequences_tested": payload.get("num_sequences_tested", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def _parse_performance_log(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out

    for key, val in re.findall(r"([A-Za-z0-9_]+):\s*([-+]?[0-9]*\.?[0-9]+)", text):
        # Ignore large non-metric parse noise by keeping plausible metric names.
        if len(key) > 80:
            continue
        out[key] = float(val)
    return out


def _collect_perf_rows(model_roots: List[Path]) -> pd.DataFrame:
    rows = []
    for root in model_roots:
        if not root.exists():
            continue
        for model_dir in root.glob("*fsdp-*"):
            if not model_dir.is_dir():
                continue
            meta = _parse_model_meta(model_dir)
            for ckpt_dir in model_dir.glob("checkpoint-*"):
                ckpt = _checkpoint_num(ckpt_dir)
                if ckpt is None:
                    continue
                perf_log = ckpt_dir / "slurm_logs" / "performance_evaluation.out"
                if not perf_log.exists():
                    continue
                metrics = _parse_performance_log(perf_log)
                if not metrics:
                    continue
                row = {
                    "model_dir": str(model_dir),
                    "checkpoint": ckpt,
                    "model_name": meta["model_name"],
                    "training_type": meta["training_type"],
                    "dataset": meta["dataset"],
                }
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


@app.command()
def main(
    mp_summary_csv: str = typer.Option(
        ..., help="Path to mp_checkpoint_summary_with_transitions_*.csv from MP dynamics script."
    ),
    model_root: List[str] = typer.Option(
        [
            "/home/mmahaut/projects/paramem/models3",
            "/home/mmahaut/projects/paramem/models2",
        ],
        "--model-root",
        help="Model roots to scan for memorization/performance checkpoint artifacts.",
    ),
    output_csv: str = typer.Option(
        "results/mp_reservoir/ft_lora_dynamics/ft_lora_mp_mem_perf_merged.csv",
        help="Output merged CSV path.",
    ),
) -> None:
    mp_df = pd.read_csv(mp_summary_csv)
    roots = [Path(p).expanduser().resolve() for p in model_root]

    mem_df = _collect_mem_rows(roots)
    perf_df = _collect_perf_rows(roots)

    merged = mp_df.copy()

    if not mem_df.empty:
        merged = merged.merge(
            mem_df,
            on=["model_dir", "model_name", "training_type", "dataset", "checkpoint"],
            how="left",
        )

    if not perf_df.empty:
        perf_cols = [
            c
            for c in perf_df.columns
            if c not in {"model_dir", "model_name", "training_type", "dataset", "checkpoint"}
        ]
        perf_df = perf_df.groupby(
            ["model_dir", "model_name", "training_type", "dataset", "checkpoint"],
            as_index=False,
        )[perf_cols].mean()
        merged = merged.merge(
            perf_df,
            on=["model_dir", "model_name", "training_type", "dataset", "checkpoint"],
            how="left",
        )

    merged = merged.sort_values(["model_name", "checkpoint"])

    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"Saved merged checkpoint table: {out_path}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")


if __name__ == "__main__":
    app()
