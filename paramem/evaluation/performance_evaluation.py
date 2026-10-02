import os
import torch
from paramem.evaluation.utils import load_model, BENCHMARK_DATASETS
from datasets import load_dataset
from typing import List, Dict, Any
import json
from pathlib import Path
from paramem.fine_tune_with_ckpts import get_dataset, get_model, get_tokenizer

"""
performance_evaluation.py

This script loads a language model from a training checkpoint (either standard or LoRA fine-tuned)
and finds ID and matrix entropy for each layer on a suite of 8 major benchmark datasets, including MMLU and others
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

def evaluate_mmlu(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on the MMLU benchmark.
    Returns:
        dict: Accuracy per subject and overall.
    """
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    # CAIS MMLU format: each sample has 'question', 'answer', and 'choices'
    for sample in dataset:  # Limit for speed; remove for full eval
        prompt = f"{sample['question']}\n"
        for idx, choice in enumerate(sample['choices']):
            prompt += f"{chr(65+idx)}) {choice}\n"
        prompt += "Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().split()[-1]
        # sample['answer'] is an integer index (0=A, 1=B, ...)
        correct_letter = chr(65 + sample['answer'])
        if pred.upper() == correct_letter:
            correct += 1
        total += 1
    return {"mmlu_accuracy": correct / total}

def evaluate_mmlu_cot(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on the MMLU benchmark with Chain-of-Thought reasoning.
    Following Llama 3.1 approach: model generates reasoning before final answer.
    Returns:
        dict: Accuracy with CoT prompting.
    """
    import re
    
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    
    for sample in dataset:
        # CoT prompt format similar to Llama 3.1 technical report
        prompt = f"Question: {sample['question']}\n\n"
        prompt += "Options:\n"
        for idx, choice in enumerate(sample['choices']):
            prompt += f"{chr(65+idx)}. {choice}\n"
        prompt += "\nLet's think step by step and then provide the answer as 'Answer: X' where X is A, B, C, or D.\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Generate longer response to capture reasoning
            output = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the full response
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        # Extract the answer - look for patterns like "Answer: A" or "The answer is A"
        # Try multiple patterns to be robust
        answer_patterns = [
            r'Answer:\s*([A-D])',
            r'answer is\s*([A-D])',
            r'correct answer is\s*([A-D])',
            r'\(([A-D])\)',
            r'([A-D])\)',
            r'^([A-D])$',
            r'\b([A-D])\b'
        ]
        
        pred_letter = None
        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                pred_letter = matches[-1].upper()  # Take the last match as final answer
                break
        
        # If no match found, try last capital letter in response
        if pred_letter is None:
            capital_letters = re.findall(r'\b([A-D])\b', response)
            if capital_letters:
                pred_letter = capital_letters[-1].upper()
        
        # Check correctness
        correct_letter = chr(65 + sample['answer'])
        if pred_letter and pred_letter == correct_letter:
            correct += 1
        total += 1
    
    return {"mmlu_cot_accuracy": correct / total}

def evaluate_test_train(checkpoint_path, split, model, tokenizer, n_samples, device, alpha, range_max) -> Dict[str, float]:
    dataset= get_dataset(checkpoint_path, n_samples=n_samples, split=split)
    dataset_compressed_name = os.path.basename(checkpoint_path)
    dataset_compressed_name = dataset_compressed_name.replace(".csv", "").replace(".txt", "")
    accuracy = 0.0
    for sample in dataset:
        prompt = sample['text']
        last_word = prompt.split()[-1]
        prompt = " ".join(prompt.split()[:-1])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().split()[-1]
        if pred == last_word:
            accuracy += 1.0
    accuracy /= len(dataset)        

    return {
        f"{dataset_compressed_name}_{split}_nwp": accuracy
    }

def evaluate_wikidata_knowledge(csv_path, model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluate model on Wikidata knowledge using proper CSV structure.
    Uses the template field and checks if answer matches any expected_answers.
    
    CSV structure:
    - template: Question template with [Y] placeholder
    - expected_answers: List of acceptable answers (stored as string repr of list)
    - query: Full context with question
    
    Returns:
        dict: Accuracy metrics
    """
    import pandas as pd
    import ast
    import re
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Sample if needed
    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
    
    correct_exact = 0
    correct_contains = 0
    total = 0
    
    for idx, row in df.iterrows():
        # Extract template and expected answers
        template = row['template']
        
        # Parse expected_answers (it's stored as string representation of a list)
        try:
            expected_answers = ast.literal_eval(row['expected_answers'])
            if not isinstance(expected_answers, list):
                expected_answers = [expected_answers]
        except:
            # Skip if we can't parse expected answers
            continue
        
        # Create prompt from template by removing [Y] placeholder
        prompt = template.replace('[Y]', '').strip()
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Generate up to 10 tokens to capture multi-word answers
            output = model.generate(
                **inputs, 
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode prediction
        pred_full = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        # Extract just the generated part (after the prompt)
        pred = pred_full[len(prompt):].strip()
        
        # Normalize prediction (lowercase, remove punctuation)
        pred_normalized = re.sub(r'[^\w\s]', '', pred.lower()).strip()
        
        # Check if prediction matches any expected answer
        match_exact = False
        match_contains = False
        
        for expected in expected_answers:
            expected_normalized = re.sub(r'[^\w\s]', '', str(expected).lower()).strip()
            
            # Exact match (after normalization)
            if pred_normalized == expected_normalized:
                match_exact = True
                match_contains = True
                break
            
            # Contains match (answer appears in prediction)
            if expected_normalized in pred_normalized or pred_normalized in expected_normalized:
                match_contains = True
        
        if match_exact:
            correct_exact += 1
        if match_contains:
            correct_contains += 1
        
        total += 1
    
    # Get dataset name
    dataset_name = os.path.basename(csv_path).replace(".csv", "")
    
    return {
        f"{dataset_name}_knowledge_exact": correct_exact / total if total > 0 else 0.0,
        f"{dataset_name}_knowledge_contains": correct_contains / total if total > 0 else 0.0
    }

def evaluate_arc(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on the ARC Challenge benchmark.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset

    correct = 0
    total = 0
    for sample in dataset:
        prompt = f"Question: {sample['question']}\nChoices: {', '.join(sample['choices']['text'])}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().split()[-1]
        if pred.lower() == sample['answerKey'].lower():
            correct += 1
        total += 1
    return {"arc_accuracy": correct / total}

def evaluate_hellaswag(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on the HellaSwag benchmark.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("hellaswag", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    for sample in dataset:
        prompt = sample['ctx'] + " " + sample['ctx_a']
        choices = sample['endings']
        scores = []
        for choice in choices:
            input_text = prompt + " " + choice
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                score = model(**inputs).logits[:, -1].mean().item()
            scores.append(score)
        pred = scores.index(max(scores))
        if pred == sample['label']:
            correct += 1
        total += 1
    return {"hellaswag_accuracy": correct / total}

def evaluate_truthfulqa(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on TruthfulQA (multiple choice).
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    for sample in dataset:
        prompt = sample['question'] + "\nAnswer:"
        choices = sample['mc1_targets']['choices']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if pred in choices:
            correct += 1
        total += 1
    return {"truthfulqa_accuracy": correct / total}

def evaluate_gsm8k(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on GSM8K (math word problems).
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    for sample in dataset:
        prompt = sample['question'] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if sample['answer'].strip() in pred:
            correct += 1
        total += 1
    return {"gsm8k_accuracy": correct / total}

def evaluate_winogrande(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on Winogrande.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("winogrande", "winogrande_xs", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    for sample in dataset:
        prompt = sample['sentence']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if pred == sample['answer']:
            correct += 1
        total += 1
    return {"winogrande_accuracy": correct / total}

def evaluate_openbookqa(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on OpenBookQA.
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("openbookqa", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    for sample in dataset:
        prompt = f"Question: {sample['question_stem']}\nChoices: {', '.join(sample['choices']['text'])}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().split()[-1]
        if pred.lower() == sample['answerKey'].lower():
            correct += 1
        total += 1
    return {"openbookqa_accuracy": correct / total}

def evaluate_lambada(model, tokenizer, n_samples, device) -> Dict[str, float]:
    """
    Evaluates the model on LAMBADA (word prediction).
    Returns:
        dict: Accuracy.
    """
    dataset = load_dataset("lambada", "plain_text", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples)) if n_samples < len(dataset) else dataset
    correct = 0
    total = 0
    for sample in dataset:
        text = sample['text']
        prompt = " ".join(text.split()[:-1])
        target = text.split()[-1]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().split()[-1]
        if pred == target:
            correct += 1
        total += 1
    return {"lambada_accuracy": correct / total}

def main():
    """
    Main function to load model and run evaluations.
    """
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoint")
    use_lora = bool(int(os.environ.get("USE_LORA", "0")))
    print(f"Parameters: checkpoint_path={checkpoint_path}, use_lora={use_lora}")

    print(f"Loading model from {checkpoint_path} (LoRA: {use_lora})...")
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

    results = {}
    print("Evaluating on benchmarks...")
    results.update(evaluate_mmlu(model, tokenizer, n_samples, device))
    results.update(evaluate_mmlu_cot(model, tokenizer, n_samples, device))
    results.update(evaluate_arc(model, tokenizer, n_samples, device))
    results.update(evaluate_hellaswag(model, tokenizer, n_samples, device))
    results.update(evaluate_truthfulqa(model, tokenizer, n_samples, device))
    results.update(evaluate_gsm8k(model, tokenizer, n_samples, device))
    # results.update(evaluate_winogrande(model, tokenizer, n_samples, device))
    results.update(evaluate_openbookqa(model, tokenizer, n_samples, device))
    results.update(evaluate_lambada(model, tokenizer, n_samples, device))
    
    # Evaluate wikidata knowledge using proper CSV structure
    wikidata_path = "/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv"
    results.update(evaluate_wikidata_knowledge(wikidata_path, model, tokenizer, n_samples, device))
    
    # Evaluate with paraphrase-based stability
    print("\nEvaluating paraphrase stability...")
    from paramem.evaluation.wikidata_paraphrase_eval import evaluate_wikidata_with_paraphrases
    
    # Set up paths for outputs
    if os.path.exists(checkpoint_path) and not as_checkpoint_0:
        slurm_logs = os.path.join(checkpoint_path, "slurm_logs")
        os.makedirs(slurm_logs, exist_ok=True)
        paraphrase_output = os.path.join(slurm_logs, "paraphrase_stability.json")
    else:
        paraphrase_output = None
    
    # Use shared paraphrase cache to avoid regenerating paraphrases for each checkpoint
    paraphrase_cache = "/home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json"
    
    results.update(evaluate_wikidata_with_paraphrases(
        wikidata_path, model, tokenizer, device,
        n_samples=min(200, n_samples),  # Use 200 samples for paraphrase eval
        num_paraphrases=5,
        use_llm_paraphrasing=True,  # Use LLM-based paraphrasing for better quality
        paraphrase_model=model,  # Use the checkpoint model itself
        paraphrase_tokenizer=tokenizer,
        output_file=paraphrase_output,
        paraphrase_cache_file=paraphrase_cache
    ))
    
    # Also do next-word prediction evaluation on training data
    for split in ["test", "train"] if "pile" in checkpoint_path or as_checkpoint_0 else ["train"]:
        results.update(evaluate_test_train(wikidata_path, split, model, tokenizer, n_samples, device, alpha=0.5, range_max=1000))
    
    data_path = "/home/mmahaut/projects/paramem/data/pile_19_short.txt"
    for split in ["test", "train"] if "pile" not in checkpoint_path or as_checkpoint_0 else ["train"]:
        results.update(evaluate_test_train(data_path, split, model, tokenizer, n_samples, device, alpha=0.5, range_max=1000))

    print("\nBenchmark Results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

    print("COMPLETED EVALUATION")

if __name__ == "__main__":
    main()