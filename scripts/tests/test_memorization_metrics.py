#!/usr/bin/env python3
"""
Quick test script for memorization metrics.
Tests on a single checkpoint with minimal data.
"""

import sys
import torch
from pathlib import Path

# Add paramem to path
sys.path.insert(0, str(Path(__file__).parent))

from paramem.memorization_metrics import compute_all_memorization_metrics
import pandas as pd


def quick_test(
    checkpoint_path: str = "models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4/checkpoint-100",
    data_file: str = "data3/wikidata_Mis7.csv"
):
    """Quick test with minimal samples."""
    
    print(f"Testing memorization metrics on: {checkpoint_path}")
    print(f"Using data from: {data_file}")
    print("="*60)
    
    # Load a small sample of training data
    df = pd.read_csv(data_file)
    
    if "query" in df.columns:
        sequences = df["query"].dropna().head(50).tolist()
    elif "text" in df.columns:
        sequences = df["text"].dropna().head(50).tolist()
    else:
        print("ERROR: Data file must have 'query' or 'text' column")
        return
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Example sequence: {sequences[0][:100]}...")
    print()
    
    # Compute metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    metrics = compute_all_memorization_metrics(
        checkpoint_path,
        sequences,
        device=device
    )
    
    # Print results
    print("\n" + "="*60)
    print("MEMORIZATION METRICS RESULTS")
    print("="*60)
    print(f"Discoverable Memorization:")
    print(f"  - Extractability Rate:    {metrics['extractability_rate']:.4f}")
    print(f"  - Avg Tokens Recovered:   {metrics['avg_tokens_recovered']:.4f}")
    print()
    print(f"Distributional Memorization:")
    print(f"  - N-gram Correlation:     {metrics['ngram_correlation']:.4f}")
    print(f"  - Comparisons Made:       {metrics['num_comparisons']}")
    print()
    print(f"Intrinsic Dimension:")
    print(f"  - ID Estimate:            {metrics['intrinsic_dimension']:.4f}")
    print()
    print(f"Sequences Tested: {metrics['num_sequences_tested']}")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test of memorization metrics")
    parser.add_argument("--checkpoint", type=str, 
                       help="Path to checkpoint (default: looks for latest)")
    parser.add_argument("--data", type=str, default="data3/wikidata_Mis7.csv",
                       help="Path to training data CSV")
    
    args = parser.parse_args()
    
    # If no checkpoint specified, try to find one
    if args.checkpoint is None:
        # Look for any checkpoint in common locations
        possible_checkpoints = [
            "models3/Llama-3.1-8B-Instruct-fsdp-1-mmlu-lr1e-4/checkpoint-100",
            "models2/checkpoint-100",
            "models/checkpoint-100",
        ]
        
        for ckpt in possible_checkpoints:
            if Path(ckpt).exists():
                args.checkpoint = ckpt
                break
        
        if args.checkpoint is None:
            print("ERROR: No checkpoint found. Please specify with --checkpoint")
            print(f"Tried: {possible_checkpoints}")
            sys.exit(1)
    
    quick_test(args.checkpoint, args.data)
