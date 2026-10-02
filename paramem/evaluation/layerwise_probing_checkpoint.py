#!/usr/bin/env python3
"""
Checkpoint-aware layerwise probing evaluation.
Called per checkpoint during training to track probing performance over time.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from datasets import load_dataset

# Add parent directory to path to import from paramem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paramem.evaluation.utils import load_model


def extract_layer_representations_simple(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,  # Reduced from 16
    max_length: int = 512
) -> Dict[int, np.ndarray]:
    """
    Extract representations from all layers of the model.
    
    Returns:
        Dictionary mapping layer index to representations array
    """
    device = next(model.parameters()).device
    all_layer_reps = {}
    
    print(f"🔍 Extracting representations from {len(texts)} samples...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Extract last token representation from each layer immediately
            # and move to CPU to free GPU memory
            for layer_idx, layer_hidden in enumerate(hidden_states):
                # Take last non-padding token representation
                last_token_reps = layer_hidden[:, -1, :].cpu().numpy()
                
                if layer_idx not in all_layer_reps:
                    all_layer_reps[layer_idx] = []
                all_layer_reps[layer_idx].append(last_token_reps)
            
            # Explicitly delete to free GPU memory
            del outputs, hidden_states
            torch.cuda.empty_cache()
    
    # Concatenate batches
    for layer_idx in all_layer_reps:
        all_layer_reps[layer_idx] = np.vstack(all_layer_reps[layer_idx])
    
    return all_layer_reps


def train_layer_probe(X: np.ndarray, y: np.ndarray, layer_idx: int) -> Dict:
    """Train a logistic regression probe on layer representations."""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train probe
    probe = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    probe.fit(X_scaled, y)
    
    # Evaluate
    y_pred = probe.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(probe, X_scaled, y, cv=5, n_jobs=-1)
    
    return {
        'layer': layer_idx,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


def load_mmlu_data(max_samples: int = 1000, split: str = "test") -> Tuple[List[str], List[str]]:
    """Load MMLU benchmark data."""
    print(f"📥 Loading MMLU {split} split...")
    dataset = load_dataset("cais/mmlu", "all", split=split)
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    
    texts = []
    labels = []
    
    for item in dataset:
        question = item['question']
        choices = item['choices']
        text = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        texts.append(text)
        labels.append(str(item['answer']))
    
    return texts, labels


def probe_checkpoint(
    checkpoint_path: str,
    use_lora: bool,
    benchmark: str = "mmlu",
    max_samples: int = 1000,
    batch_size: int = 16
):
    """
    Perform layerwise probing on a single checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        use_lora: Whether this is a LoRA checkpoint
        benchmark: Benchmark to probe on (currently only mmlu)
        max_samples: Maximum samples to use
        batch_size: Batch size for extraction
    """
    print(f"\n{'='*60}")
    print(f"🔬 Layerwise Probing Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LoRA: {use_lora}")
    print(f"Benchmark: {benchmark}")
    print(f"Max samples: {max_samples}")
    
    # Load model
    print(f"\n📦 Loading model from {checkpoint_path}...")
    try:
        model, tokenizer = load_model(checkpoint_path, use_lora)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load benchmark data
    if benchmark.lower() == "mmlu":
        texts, labels = load_mmlu_data(max_samples=max_samples)
    else:
        print(f"❌ Unsupported benchmark: {benchmark}")
        return
    
    print(f"📊 Loaded {len(texts)} samples")
    
    # Extract representations
    print(f"\n🔍 Extracting layer representations...")
    layer_reps = extract_layer_representations_simple(
        model, tokenizer, texts, batch_size=batch_size
    )
    
    print(f"✅ Extracted representations from {len(layer_reps)} layers")
    
    # Convert labels to integers
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    
    # Probe each layer
    print(f"\n🧪 Training probes on {len(layer_reps)} layers...")
    results = []
    
    for layer_idx in tqdm(sorted(layer_reps.keys()), desc="Probing layers"):
        X = layer_reps[layer_idx]
        result = train_layer_probe(X, y, layer_idx)
        results.append(result)
    
    # Save results to checkpoint's slurm_logs directory
    output_dir = Path(checkpoint_path) / "slurm_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    output_file = output_dir / f"probing_{benchmark}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved probing results to: {output_file}")
    
    # Print summary
    best_layer = df.loc[df['accuracy'].idxmax()]
    print(f"\n📊 Probing Summary ({benchmark.upper()}):")
    print(f"   Best Layer: {int(best_layer['layer'])}")
    print(f"   Best Accuracy: {best_layer['accuracy']:.4f}")
    print(f"   CV Mean: {best_layer['cv_mean']:.4f} ± {best_layer['cv_std']:.4f}")
    
    # Print completion marker for the shell script
    print("\nCOMPLETED EVALUATION")
    
    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Get parameters from environment variables (set by slurm job)
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoint")
    use_lora = int(os.environ.get("USE_LORA", "0")) == 1
    
    probe_checkpoint(
        checkpoint_path=checkpoint_path,
        use_lora=use_lora,
        benchmark="mmlu",
        max_samples=1000,  # Use smaller sample for speed during training
        batch_size=4  # Reduced batch size to avoid GPU memory issues
    )
