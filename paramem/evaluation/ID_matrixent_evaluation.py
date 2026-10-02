import os
import torch
from datasets import load_dataset
from typing import List, Dict, Any
import json
from pathlib import Path
from paramem.evaluation.utils import load_model, BENCHMARK_DATASETS
from paramem.matrix_entropy import matrix_based_entropy
from paramem.fine_tune_with_ckpts import get_dataset
from paramem.fine_tune_with_ckpts import get_model, get_tokenizer
from dadapy import Data
import numpy as np
import torch

"""
performance_evaluation.py

This script loads a language model from a training checkpoint (either standard or LoRA fine-tuned)
and evaluates its performance on a suite of 8 major benchmark datasets, including MMLU and others
commonly used for Llama2, Mistral, and ChatGPT evaluation.

Model Saving Instructions:
- Standard: Use `model.save_pretrained(checkpoint_path)` and `tokenizer.save_pretrained(checkpoint_path)`
- LoRA: Use `peft_model.save_pretrained(checkpoint_path)` and `tokenizer.save_pretrained(checkpoint_path)`

Requirements:
- transformers
- datasets
- peft (for LoRA)
- torch

Author: Matéo Mahaut
"""

def get_ID(hidden_states: Dict[str, torch.Tensor], range_max:int, alpha:float) -> Dict[str, float]:
    """
    Computes the intrinsic dimension for each layer's hidden states.
    Args:
        hidden_states (Dict[str, torch.Tensor]): Layer-wise hidden states.
    Returns:
        Dict[str, float]: Intrinsic dimension for each layer.
    """
    if not isinstance(hidden_states, dict) and (isinstance(hidden_states, tuple) or isinstance(hidden_states, list)):
        hidden_states = {f"layer_{i}": hs for i, hs in enumerate(hidden_states)}
    ids = {}
    ents = {}
    for k,v in hidden_states.items():
        # print(f"Processing layer: {k} v shape: {v.shape}")
        if v.dtype != torch.float32:
            v = v.float()
        dada_data= Data(coordinates=v.cpu().numpy())
        ids_scaling, _, _ = dada_data.return_id_scaling_gride(range_max = range_max, set_attr=True)
        ids[k] = ids_scaling
        ents[k] = matrix_based_entropy(v, alpha=alpha)
        
    return ids, ents

# List of benchmark datasets (can be expanded)
BENCHMARK_DATASETS = [
    {"name": "mmlu", "subset": "all"},  # Massive Multitask Language Understanding
    {"name": "arc", "subset": "challenge"},  # AI2 Reasoning Challenge
    {"name": "hellaswag", "subset": None},  # Commonsense reasoning
    {"name": "truthful_qa", "subset": "mc"},  # TruthfulQA multiple choice
    {"name": "gsm8k", "subset": "main"},  # Grade School Math
    {"name": "winogrande", "subset": "winogrande_xs"},  # Commonsense reasoning
    {"name": "openbookqa", "subset": "main"},  # OpenBookQA
    {"name": "lambada", "subset": "plain_text"},  # LAMBADA word prediction
]


def evaluate_mmlu(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on the MMLU benchmark.
    Returns:
        dict: Accuracy per subject and overall.
    """
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset

    # CAIS MMLU format: each sample has 'question', 'answer', and 'choices'
    all_hidden_states = {}
    for sample in dataset:
        prompt = f"{sample['question']}\n"
        for idx, choice in enumerate(sample['choices']):
            prompt += f"{chr(65+idx)}) {choice}\n"
        prompt += "Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"mmlu_intrinsic_dimension": id, "mmlu_entropy": ent}


def evaluate_arc(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on the ARC Challenge benchmark.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        prompt = f"Question: {sample['question']}\nChoices: {', '.join(sample['choices']['text'])}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)

    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"arc_intrinsic_dimension": id, "arc_entropy": ent}

def evaluate_hellaswag(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on the HellaSwag benchmark.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("hellaswag", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        prompt = sample['ctx'] + " " + sample['ctx_a']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"hellaswag_intrinsic_dimension": id, "hellaswag_entropy": ent}

def evaluate_truthfulqa(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on TruthfulQA (multiple choice).
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        prompt = sample['question'] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"truthful_qa_intrinsic_dimension": id, "truthful_qa_entropy": ent}

def evaluate_gsm8k(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on GSM8K (math word problems).
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        prompt = sample['question'] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"gsm8k_intrinsic_dimension": id, "gsm8k_entropy": ent}

def evaluate_winogrande(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on Winogrande.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("winogrande", "winogrande_xs", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        prompt = sample['sentence']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"winogrande_intrinsic_dimension": id, "winogrande_entropy": ent}

def evaluate_openbookqa(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on OpenBookQA.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("openbookqa", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        prompt = f"Question: {sample['question_stem']}\nChoices: {', '.join(sample['choices']['text'])}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"openbookqa_intrinsic_dimension": id, "openbookqa_entropy": ent}

def evaluate_lambada(model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    """
    Evaluates the model on LAMBADA (word prediction).
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("lambada", "plain_text", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    all_hidden_states = []
    for sample in dataset:
        text = sample['text']
        prompt = " ".join(text.split()[:-1])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    return {"lambada_intrinsic_dimension": id, "lambada_entropy": ent}

def evaluate_test_train(checkpoint_path, split, model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    dataset= get_dataset(checkpoint_path, n_samples=n_samples, split=split)
    dataset_compressed_name = os.path.basename(checkpoint_path)
    dataset_compressed_name = dataset_compressed_name.replace(".csv", "").replace(".txt", "")
    all_hidden_states = {}
    for sample in dataset:
        prompt = sample['text']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:,-1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:,-1, :].cpu()), dim=0)
    id, ent = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    
    return {
        f"{dataset_compressed_name}_{split}_intrinsic_dimension": id,
        f"{dataset_compressed_name}_{split}_entropy": ent
    }


def main():
    """
    Main function to load model and run evaluations.
    """
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoint")
    use_lora = bool(int(os.environ.get("USE_LORA", "0")))
    # if as_checkpoint_0, then checkpoint_path will just be model name... this causes issues for data later which we solve with tricks for now

    print(f"Loading model from {checkpoint_path} (LoRA: {use_lora})...")
    # if it's a huggingface model and not a checkpoint, load directly
    as_checkpoint_0 = False  # Initialize to False by default
    if not os.path.exists(checkpoint_path):
        as_checkpoint_0 = True # if we load directly, then it's checkpoint 0, the one from hf
        print(f"Checkpoint path {checkpoint_path} does not exist. Trying to load as HuggingFace model...")
        use_lora = False  # assume it's a full model if loading from HF hub
        model_name = checkpoint_path
        model = get_model(model_name, use_lora)
        tokenizer = get_tokenizer(model_name)

    else:
        model, tokenizer = load_model(checkpoint_path, use_lora)
    device="cuda" if model.device.type == "cuda" else "cpu"

    n_samples = 1000  # Number of samples to evaluate per benchmark
    alpha = float(os.environ.get("ALPHA", "1.0"))  # Entropy parameter
    range_max = int(os.environ.get("RANGE_MAX", "32"))  # Max
    print(f"Parameters: checkpoint_path={checkpoint_path}, use_lora={use_lora}, n_samples={n_samples}, alpha={alpha}, range_max={range_max}")
    results = {}
    print("Evaluating on benchmarks...")
    results.update(evaluate_mmlu(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_arc(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_hellaswag(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_truthfulqa(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_gsm8k(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_winogrande(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_openbookqa(model, tokenizer, n_samples, device, alpha, range_max))
    results.update(evaluate_lambada(model, tokenizer, n_samples, device, alpha, range_max))

    if not as_checkpoint_0:
        train_path = Path(checkpoint_path).parent 
        print(f"Evaluating on test/train datasets from {train_path}...")
        train_path = os.path.join(train_path, "train.csv")
        results.update(evaluate_test_train(train_path, "train", model, tokenizer, n_samples, device, alpha, range_max))
        test_path = Path(checkpoint_path).parent
        test_path = os.path.join(test_path, "test.csv")
        results.update(evaluate_test_train(test_path, "train", model, tokenizer, n_samples, device, alpha, range_max))
    if "pile" in checkpoint_path or as_checkpoint_0:
        fact_path = "/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv"
        results.update(evaluate_test_train(fact_path, "train", model, tokenizer, n_samples, device, alpha, range_max))
    elif "pile" not in checkpoint_path or as_checkpoint_0:
        pile_path="/home/mmahaut/projects/paramem/data/pile_19_short.txt"
        results.update(evaluate_test_train(pile_path, "train", model, tokenizer, n_samples, device, alpha, range_max))

    print("\nBenchmark Results:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for subk, subv in v.items():
                print(f"  {subk}: {subv:.3f}" if isinstance(subv, (float, int)) else f"  {subk}: {subv}")
        else:
            print(f"{k}: {v:.3f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    print("COMPLETED EVALUATION")

if __name__ == "__main__":
    main()