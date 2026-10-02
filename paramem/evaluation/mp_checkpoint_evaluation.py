#!/usr/bin/env python3
"""
Compute checkpoint-level MP metrics on hidden activations from a sampled set
of training sequences.

Designed to be launched by checkpoint evaluation hooks with env vars:
- CHECKPOINT_PATH
- USE_LORA (0/1)
- MODEL_NAME (optional, needed for LoRA base model)
- MP_NUM_SAMPLES (optional, default 512)
- MP_BATCH_SIZE (optional, default 8)
- MP_LAYERS (optional, comma-separated, e.g. "3,8,16")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from peft import PeftModel

from paramem.evaluation.memorization_evaluation import load_training_sequences
from paramem.fine_tune_with_ckpts import get_model, get_tokenizer
from paramem.spectral.activation_cov import compute_cov, compute_eigenspectrum
from paramem.spectral.mp_fit import fit_mp, partition


def _parse_layers(env_value: str) -> List[int]:
    if not env_value.strip():
        return []
    out = []
    for tok in env_value.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask shape [B, T], with right-padding.
    return attention_mask.sum(dim=1) - 1


def _extract_layer_activations(
    model,
    tokenizer,
    texts: List[str],
    layers: List[int],
    batch_size: int,
    device: str,
) -> Dict[int, List[np.ndarray]]:
    acts: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # includes embedding layer at index 0
            last_idx = _last_token_indices(enc.attention_mask)

            for l in layers:
                hs_idx = l + 1  # convert transformer layer index to hidden_states index
                if hs_idx >= len(hs):
                    continue
                layer_h = hs[hs_idx]  # [B, T, D]
                bsz = layer_h.shape[0]
                for b in range(bsz):
                    vec = layer_h[b, last_idx[b], :].detach().float().cpu().numpy()
                    acts[l].append(vec)

    return acts


def _compute_mp_metrics(acts: Dict[int, List[np.ndarray]]) -> List[dict]:
    rows = []
    for layer in sorted(acts.keys()):
        if len(acts[layer]) < 4:
            continue
        x = np.asarray(acts[layer], dtype=np.float64)
        n_samples, n_features = x.shape
        centered = x - x.mean(axis=0, keepdims=True)

        cov = compute_cov(x, center=True)
        evals = compute_eigenspectrum(cov, descending=False)
        gamma_eff = min(n_samples, n_features) / max(n_samples, n_features)
        mp = fit_mp(evals, gamma=gamma_eff)
        signal_mask, _ = partition(evals, mp.lambda_max)

        feature_vars = centered.var(axis=0, ddof=1)
        informative_mask = feature_vars > mp.lambda_max
        informative_indices = np.where(informative_mask)[0].astype(int).tolist()

        rows.append(
            {
                "layer": int(layer),
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "gamma_eff": float(gamma_eff),
                "sigma2": float(mp.sigma2),
                "lambda_min": float(mp.lambda_min),
                "lambda_max": float(mp.lambda_max),
                "signal_fraction": float(signal_mask.mean()),
                "num_signal": int(signal_mask.sum()),
                "num_directions": int(signal_mask.size),
                "num_informative_feature": int(informative_mask.sum()),
                "signal_fraction_feature": float(informative_mask.mean()),
                "informative_feature_indices": informative_indices,
            }
        )
    return rows


def main() -> None:
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "").strip()
    if not checkpoint_path:
        print("ERROR: CHECKPOINT_PATH environment variable not set")
        return

    use_lora = bool(int(os.environ.get("USE_LORA", "0")))
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    num_samples = int(os.environ.get("MP_NUM_SAMPLES", "512"))
    batch_size = int(os.environ.get("MP_BATCH_SIZE", "8"))
    layers_env = os.environ.get("MP_LAYERS", "3,8,16")
    requested_layers = _parse_layers(layers_env)

    cp = Path(checkpoint_path)
    out_dir = cp / "slurm_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "mp_checkpoint_metrics.json"

    # Infer dataset from checkpoint path using existing logic-compatible names.
    lower = checkpoint_path.lower()
    if "pile" in lower:
        dataset_name = "pile"
    elif "mmlu" in lower:
        dataset_name = "wikidata_Mis7"
    else:
        dataset_name = "wikidata_Mis7"

    sequences = load_training_sequences(dataset_name=dataset_name, n_samples=num_samples)
    if not sequences:
        print("ERROR: could not load sequences for MP evaluation")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_lora:
        base = get_model(model_name=model_name, use_lora=False, checkpoint_path=None, fsdp=False)
        model = PeftModel.from_pretrained(base, checkpoint_path)
        tokenizer = get_tokenizer(model_name)
    else:
        model = get_model(model_name=checkpoint_path, use_lora=False, checkpoint_path=None, fsdp=False)
        tokenizer = get_tokenizer(checkpoint_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.to(device)

    # Ensure requested layers are valid for this model depth.
    with torch.no_grad():
        probe = tokenizer([sequences[0]], return_tensors="pt", truncation=True, max_length=64).to(device)
        out = model(**probe, output_hidden_states=True)
        n_layers = len(out.hidden_states) - 1

    valid_layers = [l for l in requested_layers if 0 <= l < n_layers]
    if not valid_layers:
        # fallback: sample three roughly spaced layers
        valid_layers = sorted({0, max(0, n_layers // 2), max(0, n_layers - 1)})

    acts = _extract_layer_activations(
        model=model,
        tokenizer=tokenizer,
        texts=sequences,
        layers=valid_layers,
        batch_size=batch_size,
        device=device,
    )
    rows = _compute_mp_metrics(acts)

    payload = {
        "checkpoint_path": checkpoint_path,
        "use_lora": int(use_lora),
        "model_name": model_name,
        "num_sequences": len(sequences),
        "layers_requested": requested_layers,
        "layers_used": valid_layers,
        "metrics": rows,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("MP CHECKPOINT METRICS RESULTS")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Layers used: {valid_layers}")
    print(f"Saved: {out_json}")
    print("COMPLETED")


if __name__ == "__main__":
    main()
