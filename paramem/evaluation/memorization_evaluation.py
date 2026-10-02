#!/usr/bin/env python3
"""
memorization_evaluation.py

Evaluates memorization metrics on model checkpoints.
Follows the same structure as performance_evaluation.py and ID_matrixent_evaluation.py

Based on:
- Discoverable Memorization: https://arxiv.org/abs/2202.07646
- Distributional Memorization: https://arxiv.org/pdf/2407.14985
- Intrinsic Dimension & Memorization: https://aclanthology.org/2025.l2m2-1.2/

Author: Matéo Mahaut
"""

import os
import sys
import torch
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paramem.memorization_metrics import (
    compute_discoverable_memorization,
    compute_distributional_memorization,
    compute_embedding_intrinsic_dimension,
    get_sequence_embeddings,
    compute_train_nll_and_exposure
)
from paramem.fine_tune_with_ckpts import get_model, get_tokenizer


def load_training_sequences(dataset_name: str, n_samples: Optional[int] = None) -> List[str]:
    """Load training sequences from available datasets.

    Args:
        dataset_name: logical dataset key inferred from checkpoint path.
        n_samples: if provided and >0, cap the number of sequences; otherwise use all.
    """
    data_path = Path("/home/mmahaut/projects/paramem/data3")
    
    # Map dataset names to files
    dataset_files = {
        'wikidata_Mis7': 'wikidata_Mis7.csv',
        'wikidata_Mis7i': 'wikidata_Mis7i.csv',
        'wikidata_Qwe7': 'wikidata_Qwe7.csv',
        'wikidata_Met7': 'wikidata_Met7.csv',
        'pile': None,  # Would need different handling
    }
    
    sequences = []
    
    # First, try to load from the actual train.csv if it exists in checkpoint dir
    checkpoint_path = os.environ.get('CHECKPOINT_PATH', '')
    if checkpoint_path:
        # Go up to model directory
        model_dir = Path(checkpoint_path).parent
        train_csv = model_dir / "train.csv"
        if train_csv.exists():
            try:
                df = pd.read_csv(train_csv)
                if 'text' in df.columns:
                    seq_series = df['text'].dropna()
                    if n_samples and n_samples > 0:
                        seq_series = seq_series.head(n_samples)
                    sequences = seq_series.tolist()
                    print(f"✓ Loaded {len(sequences)} sequences from actual training data: {train_csv}")
                    return sequences
            except Exception as e:
                print(f"⚠ Could not load from {train_csv}: {e}")
    
    # Fallback: reconstruct from source CSV
    if dataset_name in dataset_files and dataset_files[dataset_name]:
        file_path = data_path / dataset_files[dataset_name]
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Reconstruct the actual training sequences (query + expected_answer)
                if 'query' in df.columns and 'expected_answers' in df.columns:
                    import random
                    rng = random.Random(42)  # Use same seed as training
                    
                    def get_complete_sequence(row):
                        try:
                            query = str(row['query']) if not pd.isna(row['query']) else ""
                            answers = row['expected_answers']
                            if pd.isna(answers):
                                return query
                            if isinstance(answers, str):
                                answers = pd.eval(answers)
                            if isinstance(answers, list) and len(answers) > 0:
                                answer = answers[rng.randint(0, len(answers) - 1)]
                                text = query + " " + str(answer)
                                # Clean quotes like in training
                                if "\" " in text[:4]:
                                    text = text.replace("\" ", "").replace(" \"", "")
                                return text
                            return query
                        except Exception as e:
                            return str(row['query']) if not pd.isna(row['query']) else ""
                    
                    df["text"] = df.apply(get_complete_sequence, axis=1)
                    seq_series = df['text'].dropna()
                    if n_samples and n_samples > 0:
                        seq_series = seq_series.head(n_samples)
                    sequences = seq_series.tolist()
                elif 'text' in df.columns:
                    seq_series = df['text'].dropna()
                    if n_samples and n_samples > 0:
                        seq_series = seq_series.head(n_samples)
                    sequences = seq_series.tolist()
                elif 'query' in df.columns:
                    seq_series = df['query'].dropna()
                    if n_samples and n_samples > 0:
                        seq_series = seq_series.head(n_samples)
                    sequences = seq_series.tolist()
                print(f"✓ Loaded {len(sequences)} sequences from {dataset_name}")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Fallback: try to load from any available wikidata file
    if not sequences:
        print(f"✗ Dataset {dataset_name} not found, trying fallback...")
        for file_name in ['wikidata_Mis7.csv', 'wikidata_Qwe7.csv', 'wikidata_Met7.csv']:
            file_path = data_path / file_name
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if 'query' in df.columns:
                        seq_series = df['query'].dropna()
                        if n_samples and n_samples > 0:
                            seq_series = seq_series.head(n_samples)
                        sequences = seq_series.tolist()
                        print(f"✓ Loaded {len(sequences)} sequences from fallback {file_name}")
                        break
                except:
                    continue
    
    return sequences


def evaluate_memorization(
    checkpoint_path: str,
    use_lora: bool = False,
    n_samples: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate memorization metrics on a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        use_lora: Whether this is a LoRA checkpoint
        n_samples: Number of sequences to test
        device: Device to use
        
    Returns:
        Dictionary of memorization metrics
    """
    print(f"\n{'='*60}")
    print(f"MEMORIZATION EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LoRA: {use_lora}")
    print(f"Samples: {n_samples if n_samples else 'ALL'}")
    print(f"{'='*60}\n")
    
    # Infer dataset from checkpoint path
    checkpoint_path_lower = checkpoint_path.lower()
    if 'wikiplus' in checkpoint_path_lower:
        dataset_name = 'wikidata_Mis7'
    elif 'pile' in checkpoint_path_lower:
        dataset_name = 'pile'
    elif 'mmlu' in checkpoint_path_lower:
        dataset_name = 'wikidata_Mis7'  # Fallback
    else:
        dataset_name = 'wikidata_Mis7'  # Default
    
    print(f"Loading training sequences from {dataset_name}...")
    sequences = load_training_sequences(dataset_name, n_samples)
    
    if not sequences:
        print("✗ ERROR: No training sequences loaded. Cannot evaluate memorization.")
        return {}
    
    print(f"✓ Loaded {len(sequences)} sequences")
    print(f"Example: {sequences[0][:100]}...\n")
    
    # Load model and tokenizer
    print(f"Loading model from {checkpoint_path}...")
    
    # Check if checkpoint has been cleaned (model files deleted but slurm_logs kept)
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        print(f"✗ ERROR: Checkpoint directory does not exist: {checkpoint_path}")
        return {}
    
    # For LoRA checkpoints, check for adapter_config.json; for full FT, check config.json
    if use_lora:
        adapter_config_path = checkpoint_path_obj / "adapter_config.json"
        if not adapter_config_path.exists():
            print(f"⚠ LoRA checkpoint was cleaned (adapter files deleted). Skipping...")
            return {}
        # For LoRA, get base model from env var MODEL_NAME or infer from checkpoint path
        base_model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        print(f"Loading base model: {base_model_name}")
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        try:
            # Load base model from hub, then attach trained adapter from checkpoint
            base_model = get_model(
                model_name=base_model_name,
                use_lora=False,
                checkpoint_path=None,
                fsdp=False
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            # Sanity: print adapter file hash to ensure per-ckpt load
            adapter_path = checkpoint_path_obj / "adapter_model.safetensors"
            if adapter_path.exists():
                import hashlib
                h = hashlib.md5()
                with open(adapter_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        h.update(chunk)
                print(f"Adapter MD5: {h.hexdigest()}")
        except Exception as e:
            print(f"✗ ERROR loading LoRA model: {e}")
            return {}
        try:
            tokenizer = get_tokenizer(base_model_name)
        except Exception as e:
            print(f"✗ ERROR loading tokenizer: {e}")
            return {}
    else:
        # Full fine-tuning checkpoint should have config.json
        config_path = checkpoint_path_obj / "config.json"
        if not config_path.exists():
            print(f"⚠ Checkpoint was cleaned (model files deleted). Skipping...")
            return {}
        try:
            model = get_model(
                model_name=checkpoint_path,
                use_lora=False,
                checkpoint_path=None,
                fsdp=False
            )
        except Exception as e:
            print(f"✗ ERROR loading model: {e}")
            return {}
        try:
            tokenizer = get_tokenizer(checkpoint_path)
        except Exception as e:
            print(f"✗ ERROR loading tokenizer: {e}")
            return {}
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"✓ Model loaded on {device}\n")
    else:
        print(f"✓ Model loaded\n")
    
    # Compute metrics
    results = {}
    
    # 1. Discoverable Memorization
    print("Computing discoverable memorization...")
    try:
        disc_metrics = compute_discoverable_memorization(
            model, tokenizer, sequences,
            context_length=10,
            suffix_length=10,
            device=device
        )
        results.update(disc_metrics)
        print(f"  ✓ Extractability rate: {disc_metrics['extractability_rate']:.4f}")
        print(f"  ✓ Avg tokens recovered: {disc_metrics['avg_tokens_recovered']:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['extractability_rate'] = 0.0
        results['avg_tokens_recovered'] = 0.0
    
    # 2. Distributional Memorization
    print("\nComputing distributional memorization...")
    try:
        dist_metrics = compute_distributional_memorization(
            model, tokenizer, sequences,
            n=3,
            device=device
        )
        results.update(dist_metrics)
        print(f"  ✓ N-gram correlation: {dist_metrics['ngram_correlation']:.4f}")
        print(f"  ✓ Comparisons: {dist_metrics['num_comparisons']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['ngram_correlation'] = 0.0
        results['num_comparisons'] = 0
    
    # 3. Intrinsic Dimension
    print("\nComputing intrinsic dimension...")
    try:
        # Use fewer sequences for ID computation (expensive)
        id_sequences = sequences[:min(50, len(sequences))]
        embeddings = get_sequence_embeddings(model, tokenizer, id_sequences, device=device)
        id_estimate = compute_embedding_intrinsic_dimension(embeddings)
        results['intrinsic_dimension'] = id_estimate
        print(f"  ✓ Intrinsic dimension: {id_estimate:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['intrinsic_dimension'] = 0.0

    # 4. Train NLL / exposure-style delta
    print("\nComputing train NLL and exposure...")
    try:
        nll_metrics = compute_train_nll_and_exposure(
            model,
            tokenizer,
            sequences,
            device=device,
            batch_size=1
        )
        results.update(nll_metrics)
        print(f"  ✓ Train NLL: {nll_metrics['train_nll']:.4f}")
        print(f"  ✓ Train PPL: {nll_metrics['train_perplexity']:.4f}")
        print(f"  ✓ Exposure delta: {nll_metrics['exposure_delta']:.4f}")
        print(f"  ✓ Exposure positive rate: {nll_metrics['exposure_positive_rate']:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['train_nll'] = 0.0
        results['train_perplexity'] = 0.0
        results['exposure_delta'] = 0.0
        results['exposure_positive_rate'] = 0.0
    
    results['num_sequences_tested'] = len(sequences)
    
    print(f"\n{'='*60}")
    print("MEMORIZATION METRICS RESULTS")
    print(f"{'='*60}")
    print(f"Extractability Rate:      {results.get('extractability_rate', 0.0):.4f}")
    print(f"Avg Tokens Recovered:     {results.get('avg_tokens_recovered', 0.0):.4f}")
    print(f"N-gram Correlation:       {results.get('ngram_correlation', 0.0):.4f}")
    print(f"Intrinsic Dimension:      {results.get('intrinsic_dimension', 0.0):.4f}")
    print(f"Train NLL:                {results.get('train_nll', 0.0):.4f}")
    print(f"Train Perplexity:         {results.get('train_perplexity', 0.0):.4f}")
    print(f"Exposure Delta:           {results.get('exposure_delta', 0.0):.4f}")
    print(f"Exposure Positive Rate:   {results.get('exposure_positive_rate', 0.0):.4f}")
    print(f"Sequences Tested:         {results.get('num_sequences_tested', 0)}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main entry point for memorization evaluation."""
    
    # Get checkpoint path from environment variable (set by launch script)
    checkpoint_path = os.environ.get('CHECKPOINT_PATH')
    use_lora_str = os.environ.get('USE_LORA', '0')
    use_lora = bool(int(use_lora_str))
    n_samples_env = os.environ.get('N_SAMPLES')
    n_samples = int(n_samples_env) if n_samples_env else None
    
    if not checkpoint_path:
        print("ERROR: CHECKPOINT_PATH environment variable not set")
        sys.exit(1)
    
    print(f"Parameters: checkpoint_path={checkpoint_path}, use_lora={use_lora}")
    
    # Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_memorization(
        checkpoint_path=checkpoint_path,
        use_lora=use_lora,
        n_samples=n_samples,
        device=device
    )
    
    # Save results as JSON (for programmatic access)
    output_dir = Path(checkpoint_path) / "slurm_logs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    json_path = output_dir / "memorization_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print("\nCOMPLETED")


if __name__ == "__main__":
    main()
