#!/usr/bin/env python3
"""
Baseline evaluation script for step 0 (non-fine-tuned) Llama-3.2-1B-Instruct model.
Runs both ID (intrinsic dimension) and performance evaluation on the base model.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
from dadapy import Data
from paramem.matrix_entropy import matrix_based_entropy
from pathlib import Path
import json

def get_ID(hidden_states, range_max=32, alpha=1.0):
    """
    Computes the intrinsic dimension for each layer's hidden states.
    Args:
        hidden_states: Layer-wise hidden states (dict or list/tuple).
        range_max: Maximum range for ID calculation.
        alpha: Alpha parameter for matrix entropy.
    Returns:
        Tuple of (ID dict, entropy dict)
    """
    if not isinstance(hidden_states, dict) and (isinstance(hidden_states, tuple) or isinstance(hidden_states, list)):
        hidden_states = {f"layer_{i}": hs for i, hs in enumerate(hidden_states)}
    
    ids = {}
    ents = {}
    for k, v in hidden_states.items():
        print(f"Processing layer: {k}, shape: {v.shape}")
        dada_data = Data(coordinates=v.cpu().numpy())
        ids_scaling, _, _ = dada_data.return_id_scaling_gride(range_max=range_max, set_attr=True)
        ids[k] = ids_scaling
        ents[k] = matrix_based_entropy(v, alpha=alpha)
        
    return ids, ents

def evaluate_mmlu_id(model, tokenizer, n_samples, device, alpha, range_max):
    """Evaluates ID and entropy on MMLU benchmark."""
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset

    all_hidden_states = {}
    correct = 0
    total = 0
    
    for sample in dataset:
        prompt = f"{sample['question']}\n"
        for idx, choice in enumerate(sample['choices']):
            prompt += f"{chr(65+idx)}) {choice}\n"
        prompt += "Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            
            # Collect hidden states for ID calculation
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:, -1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:, -1, :].cpu()), dim=0)
            
            # Performance evaluation
            generated = model.generate(**inputs, max_new_tokens=1)
            pred = tokenizer.decode(generated[0], skip_special_tokens=True).strip().split()[-1]
            correct_letter = chr(65 + sample['answer'])
            if pred.upper() == correct_letter:
                correct += 1
            total += 1

    accuracy = correct / total
    id_results, ent_results = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    
    return {
        "mmlu_intrinsic_dimension": id_results,
        "mmlu_entropy": ent_results,
        "mmlu_accuracy": accuracy
    }

def evaluate_arc_id(model, tokenizer, n_samples, device, alpha, range_max):
    """Evaluates ID and entropy on ARC Challenge benchmark."""
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    
    all_hidden_states = {}
    correct = 0
    total = 0
    
    for sample in dataset:
        prompt = f"Question: {sample['question']}\nChoices: {', '.join(sample['choices']['text'])}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            
            # Collect hidden states
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:, -1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:, -1, :].cpu()), dim=0)
            
            # Performance evaluation
            generated = model.generate(**inputs, max_new_tokens=1)
            pred = tokenizer.decode(generated[0], skip_special_tokens=True).strip().split()[-1]
            if pred.lower() == sample['answerKey'].lower():
                correct += 1
            total += 1

    accuracy = correct / total
    id_results, ent_results = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    
    return {
        "arc_intrinsic_dimension": id_results,
        "arc_entropy": ent_results,
        "arc_accuracy": accuracy
    }

def evaluate_hellaswag_id(model, tokenizer, n_samples, device, alpha, range_max):
    """Evaluates ID and entropy on HellaSwag benchmark."""
    dataset = load_dataset("hellaswag", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    
    all_hidden_states = {}
    correct = 0
    total = 0
    
    for sample in dataset:
        prompt = sample['ctx'] + " " + sample['ctx_a']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            
            # Collect hidden states
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:, -1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:, -1, :].cpu()), dim=0)
            
            # Performance evaluation (simplified - just use prompt completion)
            choices = sample['endings']
            scores = []
            for choice in choices:
                input_text = prompt + " " + choice
                choice_inputs = tokenizer(input_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    choice_output = model(**choice_inputs)
                    score = choice_output.logits[:, -1].mean().item()
                scores.append(score)
            
            pred = scores.index(max(scores))
            if pred == sample['label']:
                correct += 1
            total += 1

    accuracy = correct / total
    id_results, ent_results = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    
    return {
        "hellaswag_intrinsic_dimension": id_results,
        "hellaswag_entropy": ent_results,
        "hellaswag_accuracy": accuracy
    }

def evaluate_gsm8k_id(model, tokenizer, n_samples, device, alpha, range_max):
    """Evaluates ID and entropy on GSM8K benchmark."""
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    
    all_hidden_states = {}
    correct = 0
    total = 0
    
    for sample in dataset:
        prompt = sample['question'] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            
            # Collect hidden states
            if "layer_0" not in all_hidden_states:
                all_hidden_states = {f"layer_{i}": v[:, -1, :].cpu() for i, v in enumerate(output.hidden_states)}
            else:
                for i, v in enumerate(output.hidden_states):
                    all_hidden_states[f"layer_{i}"] = torch.cat((all_hidden_states[f"layer_{i}"], v[:, -1, :].cpu()), dim=0)
            
            # Performance evaluation
            generated = model.generate(**inputs, max_new_tokens=32)
            pred = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if sample['answer'].strip() in pred:
                correct += 1
            total += 1

    accuracy = correct / total
    id_results, ent_results = get_ID(all_hidden_states, range_max=range_max, alpha=alpha)
    
    return {
        "gsm8k_intrinsic_dimension": id_results,
        "gsm8k_entropy": ent_results,
        "gsm8k_accuracy": accuracy
    }

def main():
    """Main function to run baseline evaluation."""
    # Model configuration
    model_name = "meta-llama/Llama-3.2-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluation parameters
    n_samples = 1000  # Number of samples per benchmark
    alpha = 1.0  # Matrix entropy parameter
    range_max = 32  # Maximum range for ID calculation
    
    print(f"🚀 Starting baseline evaluation for {model_name}")
    print(f"Parameters: n_samples={n_samples}, alpha={alpha}, range_max={range_max}")
    print(f"Device: {device}")
    
    # Load base model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()
    
    # Run evaluations
    results = {}
    
    print("📊 Evaluating MMLU (ID + Performance)...")
    results.update(evaluate_mmlu_id(model, tokenizer, n_samples, device, alpha, range_max))
    
    print("📊 Evaluating ARC Challenge (ID + Performance)...")
    results.update(evaluate_arc_id(model, tokenizer, n_samples, device, alpha, range_max))
    
    print("📊 Evaluating HellaSwag (ID + Performance)...")
    results.update(evaluate_hellaswag_id(model, tokenizer, n_samples, device, alpha, range_max))
    
    print("📊 Evaluating GSM8K (ID + Performance)...")
    results.update(evaluate_gsm8k_id(model, tokenizer, n_samples, device, alpha, range_max))
    
    # Print results
    print("\n" + "="*80)
    print("🎯 BASELINE EVALUATION RESULTS (STEP 0)")
    print("="*80)
    
    for benchmark in ["mmlu", "arc", "hellaswag", "gsm8k"]:
        print(f"\n📈 {benchmark.upper()}:")
        
        # Performance results
        if f"{benchmark}_accuracy" in results:
            print(f"   Accuracy: {results[f'{benchmark}_accuracy']:.4f}")
        
        # ID results
        if f"{benchmark}_intrinsic_dimension" in results:
            id_data = results[f"{benchmark}_intrinsic_dimension"]
            if isinstance(id_data, dict) and id_data:
                mean_id = np.mean([v[-1] if isinstance(v, list) else v for v in id_data.values() if v])
                print(f"   Mean ID: {mean_id:.4f}")
                print(f"   Layer ID range: {min([v[-1] if isinstance(v, list) else v for v in id_data.values() if v]):.4f} → {max([v[-1] if isinstance(v, list) else v for v in id_data.values() if v]):.4f}")
        
        # Entropy results  
        if f"{benchmark}_entropy" in results:
            ent_data = results[f"{benchmark}_entropy"]
            if isinstance(ent_data, dict) and ent_data:
                mean_ent = np.mean([v for v in ent_data.values() if not np.isnan(v)])
                print(f"   Mean Entropy: {mean_ent:.4f}")
    
    # Save results to file
    output_dir = "/home/mmahaut/projects/paramem/step0_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(output_dir, "baseline_results.json")
    with open(output_file, 'w') as f:
        # Convert numpy arrays/floats to regular Python types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {str(subk): (subv.tolist() if hasattr(subv, 'tolist') else float(subv)) 
                                 for subk, subv in v.items()}
            else:
                json_results[k] = float(v) if hasattr(v, 'item') else v
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎉 Baseline evaluation completed!")

if __name__ == "__main__":
    main()