#!/usr/bin/env python3
"""
Generate paraphrase cache using distributed processing.
Each worker processes a subset of questions.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def generate_paraphrases_with_llm(
    question: str,
    num_paraphrases: int,
    model,
    tokenizer,
    temperature: float = 0.7
) -> List[str]:
    """Generate paraphrases using LLM."""
    paraphrases = []
    
    prompt = f"""Paraphrase the following question in {num_paraphrases} different ways. Keep the meaning exactly the same but change the wording.

Original: {question}

Provide {num_paraphrases} paraphrases, one per line:
1."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse out the paraphrases
    lines = generated.split('\n')
    for line in lines:
        # Look for numbered lines
        match = re.match(r'^\d+\.\s*(.+)$', line.strip())
        if match:
            paraphrases.append(match.group(1).strip())
            if len(paraphrases) >= num_paraphrases:
                break
    
    # If we didn't get enough, pad with the original
    while len(paraphrases) < num_paraphrases:
        paraphrases.append(question)
    
    return paraphrases[:num_paraphrases]


def main():
    parser = argparse.ArgumentParser(description='Generate paraphrase cache')
    parser.add_argument('--rank', type=int, required=True, help='Worker rank (0-indexed)')
    parser.add_argument('--world-size', type=int, required=True, help='Total number of workers')
    parser.add_argument('--wikidata-path', type=str, required=True, help='Path to Wikidata CSV')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for partial caches')
    parser.add_argument('--n-samples', type=int, default=200, help='Total number of questions')
    parser.add_argument('--num-paraphrases', type=int, default=5, help='Paraphrases per question')
    
    args = parser.parse_args()
    
    print(f"Worker {args.rank}/{args.world_size} starting...")
    print(f"Output dir: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading base model: meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on device: {model.device}")
    
    # Load questions
    print(f"Loading questions from: {args.wikidata_path}")
    df = pd.read_csv(args.wikidata_path)
    df = df.head(args.n_samples)
    
    # Split questions among workers
    questions = df['template'].tolist()  # Use 'template' column
    total_questions = len(questions)
    
    # Calculate this worker's slice
    questions_per_worker = (total_questions + args.world_size - 1) // args.world_size
    start_idx = args.rank * questions_per_worker
    end_idx = min(start_idx + questions_per_worker, total_questions)
    
    my_questions = questions[start_idx:end_idx]
    
    print(f"Worker {args.rank}: Processing questions {start_idx} to {end_idx-1} ({len(my_questions)} questions)")
    
    # Generate paraphrases
    cache = {}
    for i, question in enumerate(my_questions):
        global_idx = start_idx + i
        print(f"Worker {args.rank}: [{i+1}/{len(my_questions)}] Question {global_idx}: {question[:50]}...")
        
        paraphrases = generate_paraphrases_with_llm(
            question,
            args.num_paraphrases,
            model,
            tokenizer
        )
        
        cache[question] = paraphrases
        
        # Save progress periodically
        if (i + 1) % 10 == 0:
            partial_cache_path = os.path.join(args.output_dir, f'paraphrase_cache_worker_{args.rank}.json')
            with open(partial_cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"Worker {args.rank}: Saved progress ({i+1}/{len(my_questions)} questions)")
    
    # Save final results
    output_path = os.path.join(args.output_dir, f'paraphrase_cache_worker_{args.rank}.json')
    with open(output_path, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Worker {args.rank}: Complete! Saved {len(cache)} questions to {output_path}")


if __name__ == '__main__':
    main()
