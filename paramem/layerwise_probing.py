#!/usr/bin/env python3
"""
Experiment 2: Layerwise probing tasks for Full Fine-tuning, LoRA, and Original models.
Tests MMLU and other benchmark performance using single-layer representations.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import logging

app = typer.Typer()


def load_benchmark_from_hf(benchmark_name: str, split: str = "test", max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load a benchmark dataset from Hugging Face.
    
    Args:
        benchmark_name: Name of the benchmark (mmlu, arc, hellaswag, etc.)
        split: Dataset split to use
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    try:
        if benchmark_name.lower() == "mmlu":
            # MMLU has multiple subjects, we'll use a subset
            dataset = load_dataset("cais/mmlu", "all", split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                # MMLU format: question + 4 choices
                question = item['question']
                choices = item['choices']
                text = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
                texts.append(text)
                labels.append(str(item['answer']))
                
        elif benchmark_name.lower() == "arc":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                question = item['question']
                choices = item['choices']['text']
                text = f"{question}\n" + "\n".join([f"{i+1}) {c}" for i, c in enumerate(choices)])
                texts.append(text)
                labels.append(item['answerKey'])
                
        elif benchmark_name.lower() == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split="validation" if split == "test" else split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                ctx = item['ctx']
                text = f"{ctx}"
                texts.append(text)
                labels.append(str(item['label']))
                
        elif benchmark_name.lower() == "winogrande":
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation" if split == "test" else split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                sentence = item['sentence']
                option1 = item['option1']
                option2 = item['option2']
                text = f"{sentence}\n1) {option1}\n2) {option2}"
                texts.append(text)
                labels.append(str(item['answer']))
                
        elif benchmark_name.lower() == "truthful_qa":
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                question = item['question']
                text = question
                texts.append(text)
                # Use best answer as label
                labels.append(str(item['mc1_targets']['labels'].index(1) if 1 in item['mc1_targets']['labels'] else 0))
                
        elif benchmark_name.lower() == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                question = item['question']
                texts.append(question)
                # Extract numeric answer
                answer = item['answer'].split('####')[-1].strip()
                labels.append(answer)
                
        elif benchmark_name.lower() == "openbookqa":
            dataset = load_dataset("openbookqa", "main", split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                question = item['question_stem']
                choices = item['choices']['text']
                text = f"{question}\n" + "\n".join([f"{i+1}) {c}" for i, c in enumerate(choices)])
                texts.append(text)
                labels.append(item['answerKey'])
                
        elif benchmark_name.lower() == "lambada":
            dataset = load_dataset("lambada", split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for item in dataset:
                text = item['text']
                texts.append(text)
                # Lambada predicts the last word
                last_word = text.split()[-1]
                labels.append(last_word)
        
        else:
            print(f"⚠️  Unknown benchmark: {benchmark_name}")
            return [], []
            
        print(f"✅ Loaded {len(texts)} samples from {benchmark_name}")
        return texts, labels
        
    except Exception as e:
        print(f"❌ Error loading {benchmark_name}: {e}")
        return [], []


def load_pickle(file_path: str) -> Dict:
    """Load a pickle file containing layer representations."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_representations_from_model(
    model_path: str,
    tokenizer_name: str,
    data_file: str,
    batch_size: int = 16,
    max_samples: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """
    Extract layer-wise representations from a model on given data.
    
    Args:
        model_path: Path to the model checkpoint
        tokenizer_name: Name/path of the tokenizer
        data_file: TSV file with format: id\\tlabel\\ttext
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
        
    Returns:
        Dictionary mapping layer indices to representations
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    # Load model
    print(f"📥 Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print(f"📥 Loading data from {data_file}")
    inputs = []
    labels = []
    
    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                labels.append(parts[1])
                inputs.append(parts[2])
    
    if max_samples:
        inputs = inputs[:max_samples]
        labels = labels[:max_samples]
    
    print(f"📊 Processing {len(inputs)} samples")
    
    # Extract representations
    states = {}
    
    for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting representations"):
        batch_inputs = inputs[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(device)
        
        # Get last true token indices
        last_token_indices = []
        for att_mask in encoded.attention_mask:
            if 0 not in att_mask:
                last_token_indices.append(len(att_mask) - 1)
            else:
                last_token_indices.append(att_mask.tolist().index(0) - 1)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # Extract last token activations per layer
        for layer_idx, layer_hidden in enumerate(hidden_states):
            if layer_idx not in states:
                states[layer_idx] = []
            
            for batch_idx, token_idx in enumerate(last_token_indices):
                activation = layer_hidden[batch_idx, token_idx].cpu().numpy()
                states[layer_idx].append(activation)
    
    return states, labels


def train_layer_probe(
    representations: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Train a linear probe on representations from a single layer.
    
    Args:
        representations: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,)
        layer_idx: Layer index for logging
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression probe
    probe = LogisticRegression(max_iter=1000, random_state=random_state, multi_class='multinomial')
    probe.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = probe.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    return {
        'layer': layer_idx,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': len(np.unique(labels))
    }


def probe_all_layers(
    representations_dict: Dict[int, List],
    labels: List[str],
    model_name: str,
    output_dir: Path
) -> pd.DataFrame:
    """
    Train probes on all layers and return results.
    
    Args:
        representations_dict: Dictionary mapping layer indices to representations
        labels: List of labels
        model_name: Name of the model
        output_dir: Directory to save results
        
    Returns:
        DataFrame with results for all layers
    """
    # Convert string labels to integers
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    
    results = []
    
    print(f"\n🧪 Training probes for {model_name}")
    print(f"📊 Total samples: {len(labels)}, Classes: {len(unique_labels)}")
    
    for layer_idx in tqdm(sorted(representations_dict.keys()), desc="Training layer probes"):
        # Convert representations to numpy array
        X = np.array(representations_dict[layer_idx])
        
        # Train probe
        result = train_layer_probe(X, y, layer_idx)
        result['model'] = model_name
        results.append(result)
        
        tqdm.write(f"Layer {layer_idx}: Acc={result['accuracy']:.4f}, CV={result['cv_mean']:.4f}±{result['cv_std']:.4f}")
    
    df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"probing_results_{model_name}.csv", index=False)
    
    return df


def plot_probing_results(
    results_dict: Dict[str, pd.DataFrame],
    benchmark_name: str,
    output_file: Path
):
    """
    Plot probing results across layers for different models.
    
    Args:
        results_dict: Dictionary mapping model names to result DataFrames
        benchmark_name: Name of the benchmark
        output_file: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Plot accuracy curves
    for model_name, df in results_dict.items():
        # Ensure layers are sorted numerically
        df_sorted = df.sort_values('layer')
        
        # Plot test accuracy as main line
        plt.plot(df_sorted['layer'], df_sorted['accuracy'], marker='o', linewidth=2, 
                markersize=6, label=f"{model_name} (Test)", alpha=0.8)
        
        # Plot CV mean as dashed line with std bands
        line = plt.plot(df_sorted['layer'], df_sorted['cv_mean'], linestyle='--', 
                       linewidth=1, alpha=0.5)[0]
        plt.fill_between(df_sorted['layer'], 
                         df_sorted['cv_mean'] - df_sorted['cv_std'], 
                         df_sorted['cv_mean'] + df_sorted['cv_std'], 
                         alpha=0.15, color=line.get_color())
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Layerwise Probing Results - {benchmark_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved probing plot: {output_file}")


def plot_heatmap_comparison(
    results_dict: Dict[str, pd.DataFrame],
    benchmark_name: str,
    output_file: Path
):
    """
    Plot heatmap comparing probing accuracy across models and layers.
    
    Args:
        results_dict: Dictionary mapping model names to result DataFrames
        benchmark_name: Name of the benchmark
        output_file: Path to save the plot
    """
    # Create matrix: rows = models, columns = layers
    model_names = list(results_dict.keys())
    
    # Get all layer indices
    all_layers = set()
    for df in results_dict.values():
        all_layers.update(df['layer'].values)
    all_layers = sorted(all_layers)
    
    # Build matrix
    matrix = np.zeros((len(model_names), len(all_layers)))
    
    for i, model_name in enumerate(model_names):
        df = results_dict[model_name]
        for j, layer in enumerate(all_layers):
            layer_data = df[df['layer'] == layer]
            if not layer_data.empty:
                matrix[i, j] = layer_data['accuracy'].values[0]
    
    # Plot
    plt.figure(figsize=(16, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[f"L{i}" for i in all_layers],
                yticklabels=model_names,
                cbar_kws={'label': 'Accuracy'})
    plt.title(f'Layerwise Probing Accuracy Heatmap - {benchmark_name}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved heatmap: {output_file}")


def plot_best_layer_comparison(
    results_dict: Dict[str, pd.DataFrame],
    benchmark_name: str,
    output_file: Path
):
    """
    Plot bar chart comparing best layer accuracy for each model.
    
    Args:
        results_dict: Dictionary mapping model names to result DataFrames
        benchmark_name: Name of the benchmark
        output_file: Path to save the plot
    """
    best_results = []
    
    for model_name, df in results_dict.items():
        best_idx = df['accuracy'].idxmax()
        best_row = df.iloc[best_idx]
        best_results.append({
            'Model': model_name,
            'Best Layer': int(best_row['layer']),
            'Accuracy': best_row['accuracy'],
            'CV Mean': best_row['cv_mean'],
            'CV Std': best_row['cv_std']
        })
    
    df_best = pd.DataFrame(best_results)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of accuracies
    ax1.bar(df_best['Model'], df_best['Accuracy'], color='skyblue', alpha=0.8)
    ax1.errorbar(df_best['Model'], df_best['CV Mean'], 
                 yerr=df_best['CV Std'], fmt='o', color='red', 
                 capsize=5, label='CV Mean ± Std')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Best Layer Performance - {benchmark_name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bar chart of best layers
    ax2.bar(df_best['Model'], df_best['Best Layer'], color='lightcoral', alpha=0.8)
    ax2.set_ylabel('Layer Index', fontsize=12)
    ax2.set_title(f'Best Performing Layer - {benchmark_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved best layer comparison: {output_file}")


@app.command()
def main(
    models_dir: str = "/home/mmahaut/projects/paramem/models2",
    benchmark_data_dir: str = "/home/mmahaut/projects/paramem/benchmark/",
    output_dir: str = "/home/mmahaut/projects/paramem/output/layerwise_probing",
    dataset: str = "pile",  # pile or wikiplus
    benchmarks: str = "mmlu,arc,hellaswag,gsm8k",  # Comma-separated list
    max_samples: Optional[int] = 2000,
    batch_size: int = 16
):
    """
    Perform layerwise probing on Full FT, LoRA FT, and Original models.
    Extracts representations on-the-fly from model checkpoints.
    
    Args:
        models_dir: Directory containing model checkpoints (e.g., models2/)
        benchmark_data_dir: Directory containing benchmark data files
        output_dir: Directory to save results
        dataset: Dataset to use (pile or wikiplus)
        benchmarks: List of benchmarks to probe
        max_samples: Maximum samples per benchmark
        batch_size: Batch size for extraction
    """
    from extract_representations_utils import extract_layer_representations, load_benchmark_data
    
    # Parse benchmarks from comma-separated string
    benchmark_list = [b.strip() for b in benchmarks.split(',')]
    
    benchmark_path = Path(benchmark_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find model checkpoints - ONLY use 8B models
    models_path = Path(models_dir)
    base_model = "meta-llama/Llama-3.1-8B-Instruct"  # Force 8B
    
    full_ft_path = None
    lora_ft_path = None
    
    # Only look for 8B models
    for model_dir in models_path.iterdir():
        if not model_dir.is_dir() or "Llama" not in model_dir.name:
            continue
        # Skip if not matching dataset or not 8B
        if dataset.lower() not in model_dir.name.lower() or "8B" not in model_dir.name:
            continue
        
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
        
        if "fsdp-0" in model_dir.name and full_ft_path is None:
            checkpoints = sorted([d for d in model_dir.iterdir() if d.name.startswith("checkpoint-")])
            if checkpoints:
                full_ft_path = checkpoints[-1]
        elif "fsdp-1" in model_dir.name and lora_ft_path is None:
            checkpoints = sorted([d for d in model_dir.iterdir() if d.name.startswith("checkpoint-")])
            if checkpoints:
                lora_ft_path = checkpoints[-1]
    
    print(f"\n🔍 Found model paths for {dataset}:")
    print(f"  Original: {base_model}")
    print(f"  Full FT: {full_ft_path}")
    print(f"  LoRA FT: {lora_ft_path}")
    
    # Prepare to extract representations
    representations = {}
    model_names = {}
    # Find LoRA adapter path (try both .safetensors and .bin)
    lora_adapter_path = None
    if lora_ft_path:
        if (lora_ft_path / "adapter_model.safetensors").exists():
            lora_adapter_path = str(lora_ft_path)
        elif (lora_ft_path / "adapter_model.bin").exists():
            lora_adapter_path = str(lora_ft_path)
    
    model_paths = {
        'original': (base_model, False),
        'full_ft': (str(full_ft_path) if full_ft_path else None, False),
        'lora_ft': (lora_adapter_path, True)
    }
    model_names = {
        'original': "Original",
        'full_ft': f"Full-FT-{dataset}",
        'lora_ft': f"LoRA-FT-{dataset}"
    }
    
    # Process each benchmark
    for benchmark in benchmark_list:
        print(f"\n{'='*60}")
        print(f"📊 Processing benchmark: {benchmark.upper()}")
        print(f"{'='*60}")
        
        # Try to load from Hugging Face first
        texts, labels = load_benchmark_from_hf(benchmark, split="test", max_samples=max_samples)
        
        # If HF loading failed, try local files
        if len(texts) == 0:
            print(f"📁 Trying to load from local file...")
            benchmark_file = None
            for ext in ['.jsonl', '.tsv', '.csv', '.txt']:
                potential_file = benchmark_path / f"{benchmark}{ext}"
                if potential_file.exists():
                    benchmark_file = potential_file
                    break
                # Also try dev/test splits
                for split in ['dev', 'test', 'val']:
                    potential_file = benchmark_path / f"{benchmark}_{split}{ext}"
                    if potential_file.exists():
                        benchmark_file = potential_file
                        break
            
            if benchmark_file is None:
                print(f"⚠️  Warning: Data not found for {benchmark} (tried HF and local), skipping...")
                continue
            
            print(f"📥 Using benchmark data: {benchmark_file}")
            texts, labels = load_benchmark_data(str(benchmark_file), max_samples=max_samples)
        
        print(f"📊 Using {len(texts)} samples for {benchmark}")
        
        # Extract representations and probe each model
        all_results = {}
        
        for model_type, (model_path, is_lora) in model_paths.items():
            if model_path is None:
                print(f"⚠️  Skipping {model_type} (not found)")
                continue
                
            print(f"\n🔬 Extracting and probing {model_type} on {benchmark}")
            
            # Extract representations
            reps = extract_layer_representations(
                model_path,
                texts,
                batch_size=batch_size,
                is_lora=is_lora,
                base_model_name=base_model
            )
            
            # Probe all layers
            df_results = probe_all_layers(
                reps,
                labels,
                model_names[model_type],
                output_path / benchmark
            )
            
            all_results[model_names[model_type]] = df_results
        
        # Plot results
        if all_results:
            plot_probing_results(
                all_results,
                benchmark.upper(),
                output_path / f"probing_curves_{benchmark}_{dataset}.png"
            )
            
            plot_heatmap_comparison(
                all_results,
                benchmark.upper(),
                output_path / f"probing_heatmap_{benchmark}_{dataset}.png"
            )
            
            plot_best_layer_comparison(
                all_results,
                benchmark.upper(),
                output_path / f"probing_best_{benchmark}_{dataset}.png"
            )
    
    print(f"\n{'='*60}")
    print(f"✅ All probing experiments completed!")
    print(f"📁 Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    app()
