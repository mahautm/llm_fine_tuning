#!/usr/bin/env python3
"""
Paraphrase-based stability evaluation for Wikidata knowledge.
Inspired by: https://github.com/amazon-science/factual-confidence-of-llms

Pipeline:
1. Load Wikidata questions from CSV
2. Generate paraphrases of each question template
3. Evaluate model on original + all paraphrases
4. Measure consistency and accuracy across paraphrases

Metrics:
- per_question_consistency: How often paraphrases give the same answer
- per_question_accuracy: How often answers are correct
- overall_stability: Average consistency across all questions
- accuracy_when_consistent: Accuracy when model is consistent
"""

import os
import torch
import pandas as pd
import numpy as np
import ast
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import json


def generate_paraphrases_with_llm(
    question: str,
    num_paraphrases: int = 5,
    temperature: float = 0.7,
    paraphrase_model=None,
    paraphrase_tokenizer=None
) -> List[str]:
    """
    Generate paraphrases of a question using an LLM.
    
    Args:
        question: Original question template
        num_paraphrases: Number of paraphrases to generate
        temperature: Sampling temperature for diversity
        paraphrase_model: Model to use for paraphrasing (if None, uses simple variations)
        paraphrase_tokenizer: Tokenizer for paraphrase model
    
    Returns:
        List of paraphrased questions
    """
    paraphrases = []
    
    if paraphrase_model is not None and paraphrase_tokenizer is not None:
        # Use LLM to generate paraphrases
        prompt = f"""Paraphrase the following question in {num_paraphrases} different ways. Keep the meaning exactly the same but change the wording.

Original: {question}

Provide {num_paraphrases} paraphrases, one per line:
1."""
        
        inputs = paraphrase_tokenizer(prompt, return_tensors="pt").to(paraphrase_model.device)
        
        with torch.no_grad():
            outputs = paraphrase_model.generate(
                **inputs,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True
            )
        
        generated = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse out the paraphrases
        lines = generated.split('\n')
        for line in lines:
            # Look for numbered lines
            match = re.match(r'^\d+\.\s*(.+)$', line.strip())
            if match:
                paraphrases.append(match.group(1).strip())
                if len(paraphrases) >= num_paraphrases:
                    break
    
    else:
        # Fallback: Generate simple rule-based paraphrases
        paraphrases = generate_rule_based_paraphrases(question, num_paraphrases)
    
    return paraphrases[:num_paraphrases]


def generate_rule_based_paraphrases(question: str, num_paraphrases: int = 5) -> List[str]:
    """
    Generate simple rule-based paraphrases by varying question structure.
    
    Examples:
    - "X can be categorized as a type of Y" 
      → "X is a type of Y"
      → "X belongs to the category Y"
      → "The type of X is Y"
    """
    paraphrases = []
    
    # Common patterns and their variations
    patterns = [
        # Pattern 1: "X can be categorized as a type of"
        (r'(.+) can be categorized as a type of$', [
            r'\1 is a type of',
            r'\1 belongs to the category',
            r'The type of \1 is',
            r'\1 is classified as',
            r'\1 falls under the category of'
        ]),
        # Pattern 2: "The X of Y is"
        (r'The (.+) of (.+) is$', [
            r"What is the \1 of \2?",
            r"\2's \1 is",
            r"The \1 that \2 has is",
            r"For \2, the \1 is",
            r"\2 has a \1 of"
        ]),
        # Pattern 3: "X was created by"
        (r'(.+) was created by$', [
            r'The creator of \1 is',
            r'\1 was made by',
            r'Who created \1?',
            r'The author of \1 is',
            r'\1 is the work of'
        ]),
        # Pattern 4: "X is located in"
        (r'(.+) is located in$', [
            r'The location of \1 is',
            r'\1 can be found in',
            r'Where is \1 located?',
            r'\1 is situated in',
            r'You can find \1 in'
        ])
    ]
    
    # Try to match patterns and generate variations
    for pattern, variations in patterns:
        match = re.search(pattern, question.strip())
        if match:
            for variation_pattern in variations[:num_paraphrases]:
                try:
                    paraphrase = re.sub(pattern, variation_pattern, question.strip())
                    paraphrases.append(paraphrase)
                except:
                    continue
            break
    
    # If no pattern matched or not enough paraphrases, add simple variations
    while len(paraphrases) < num_paraphrases:
        if not paraphrases:
            # Just add the original if we couldn't generate any
            paraphrases.append(question)
        else:
            # Duplicate some paraphrases
            paraphrases.append(paraphrases[len(paraphrases) % len(paraphrases)])
    
    return paraphrases[:num_paraphrases]


def evaluate_question_with_paraphrases(
    question: str,
    paraphrases: List[str],
    expected_answers: List[str],
    model,
    tokenizer,
    device: str
) -> Dict:
    """
    Evaluate model on original question and all paraphrases.
    
    Returns:
        Dictionary with:
        - predictions: List of predictions for [original] + paraphrases
        - consistency: Whether all predictions are the same
        - accuracy: Fraction of predictions that match expected_answers
        - correct_answers: Which predictions matched
    """
    all_questions = [question] + paraphrases
    predictions = []
    correct = []
    
    for q in all_questions:
        # Generate answer
        inputs = tokenizer(q, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        pred_full = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        pred = pred_full[len(q):].strip()
        
        # Normalize
        pred_normalized = re.sub(r'[^\w\s]', '', pred.lower()).strip()
        predictions.append(pred_normalized)
        
        # Check if correct
        is_correct = False
        for expected in expected_answers:
            expected_normalized = re.sub(r'[^\w\s]', '', str(expected).lower()).strip()
            if pred_normalized == expected_normalized or \
               expected_normalized in pred_normalized or \
               pred_normalized in expected_normalized:
                is_correct = True
                break
        
        correct.append(is_correct)
    
    # Check consistency: all predictions should be the same
    unique_predictions = set(predictions)
    is_consistent = len(unique_predictions) == 1
    
    # Calculate metrics
    accuracy = sum(correct) / len(correct) if correct else 0.0
    
    return {
        'predictions': predictions,
        'is_consistent': is_consistent,
        'accuracy': accuracy,
        'correct': correct,
        'num_correct': sum(correct),
        'total': len(correct)
    }


def evaluate_wikidata_with_paraphrases(
    csv_path: str,
    model,
    tokenizer,
    device: str,
    n_samples: int = 1000,
    num_paraphrases: int = 5,
    use_llm_paraphrasing: bool = False,
    paraphrase_model=None,
    paraphrase_tokenizer=None,
    output_file: Optional[str] = None,
    paraphrase_cache_file: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate model's factual knowledge with paraphrase-based stability testing.
    
    Args:
        csv_path: Path to Wikidata CSV
        model: Model to evaluate
        tokenizer: Tokenizer
        device: Device to run on
        n_samples: Number of questions to evaluate
        num_paraphrases: Number of paraphrases per question
        use_llm_paraphrasing: Whether to use LLM for paraphrasing (vs rule-based)
        paraphrase_model: Model for generating paraphrases (optional)
        paraphrase_tokenizer: Tokenizer for paraphrase model (optional)
        output_file: Path to save detailed results (optional)
        paraphrase_cache_file: Path to load/save paraphrase cache (optional)
    
    Returns:
        Dictionary of metrics
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load paraphrase cache if exists
    paraphrase_cache = {}
    if paraphrase_cache_file and os.path.exists(paraphrase_cache_file):
        print(f"Loading cached paraphrases from {paraphrase_cache_file}...")
        try:
            with open(paraphrase_cache_file, 'r') as f:
                paraphrase_cache = json.load(f)
            print(f"✅ Loaded {len(paraphrase_cache)} cached paraphrase sets")
        except Exception as e:
            print(f"⚠️  Could not load cache: {e}")
            paraphrase_cache = {}
    
    # Sample if needed
    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
    
    results_per_question = []
    paraphrases_to_save = {}  # Track new paraphrases to add to cache
    
    print(f"\n{'='*60}")
    print(f"Paraphrase-Based Stability Evaluation")
    print(f"{'='*60}")
    print(f"Questions: {len(df)}")
    print(f"Paraphrases per question: {num_paraphrases}")
    print(f"Paraphrasing method: {'LLM' if use_llm_paraphrasing else 'Rule-based'}")
    print(f"Cache available: {len(paraphrase_cache)} questions")
    print(f"{'='*60}\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating with paraphrases"):
        # Extract question and expected answers
        template = row['template']
        
        try:
            expected_answers = ast.literal_eval(row['expected_answers'])
            if not isinstance(expected_answers, list):
                expected_answers = [expected_answers]
        except:
            continue
        
        # Create question from template
        question = template.replace('[Y]', '').strip()
        
        # Try to use cached paraphrases
        cache_key = question
        if cache_key in paraphrase_cache:
            paraphrases = paraphrase_cache[cache_key]
        else:
            # Generate new paraphrases
            if use_llm_paraphrasing and paraphrase_model is not None:
                paraphrases = generate_paraphrases_with_llm(
                    question, 
                    num_paraphrases,
                    paraphrase_model=paraphrase_model,
                    paraphrase_tokenizer=paraphrase_tokenizer
                )
            else:
                paraphrases = generate_rule_based_paraphrases(question, num_paraphrases)
            
            # Mark for saving to cache
            paraphrases_to_save[cache_key] = paraphrases
        
        # Evaluate with paraphrases
        result = evaluate_question_with_paraphrases(
            question,
            paraphrases,
            expected_answers,
            model,
            tokenizer,
            device
        )
        
        result['question'] = question
        result['expected_answers'] = expected_answers
        result['paraphrases'] = paraphrases
        
        results_per_question.append(result)
    
    # Aggregate metrics
    total_questions = len(results_per_question)
    consistent_questions = sum(1 for r in results_per_question if r['is_consistent'])
    
    # Average accuracy across all questions and paraphrases
    overall_accuracy = np.mean([r['accuracy'] for r in results_per_question])
    
    # Consistency rate
    consistency_rate = consistent_questions / total_questions if total_questions > 0 else 0.0
    
    # Accuracy when model is consistent
    consistent_results = [r for r in results_per_question if r['is_consistent']]
    accuracy_when_consistent = np.mean([r['accuracy'] for r in consistent_results]) if consistent_results else 0.0
    
    # Accuracy when model is inconsistent
    inconsistent_results = [r for r in results_per_question if not r['is_consistent']]
    accuracy_when_inconsistent = np.mean([r['accuracy'] for r in inconsistent_results]) if inconsistent_results else 0.0
    
    metrics = {
        'wikidata_paraphrase_overall_accuracy': overall_accuracy,
        'wikidata_paraphrase_consistency_rate': consistency_rate,
        'wikidata_paraphrase_acc_when_consistent': accuracy_when_consistent,
        'wikidata_paraphrase_acc_when_inconsistent': accuracy_when_inconsistent,
        'wikidata_paraphrase_num_consistent': consistent_questions,
        'wikidata_paraphrase_num_questions': total_questions
    }
    
    # Save paraphrase cache if we generated new paraphrases
    if paraphrase_cache_file and paraphrases_to_save:
        # Merge with existing cache
        paraphrase_cache.update(paraphrases_to_save)
        try:
            cache_path = Path(paraphrase_cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(paraphrase_cache_file, 'w') as f:
                json.dump(paraphrase_cache, f, indent=2)
            print(f"✅ Saved {len(paraphrases_to_save)} new paraphrases to cache (total: {len(paraphrase_cache)})")
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")
    
    # Save detailed results if requested
    if output_file:
        output_data = {
            'metrics': metrics,
            'per_question_results': results_per_question
        }
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Detailed results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Paraphrase Stability Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Consistency Rate: {consistency_rate:.3f} ({consistent_questions}/{total_questions})")
    print(f"Accuracy when Consistent: {accuracy_when_consistent:.3f}")
    print(f"Accuracy when Inconsistent: {accuracy_when_inconsistent:.3f}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    """
    Example usage:
    
    from paramem.evaluation.utils import load_model
    
    checkpoint_path = "path/to/checkpoint"
    model, tokenizer = load_model(checkpoint_path, use_lora=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    metrics = evaluate_wikidata_with_paraphrases(
        csv_path="/home/mmahaut/projects/paramem/data3/wikidata_Mis7.csv",
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_samples=100,
        num_paraphrases=5,
        use_llm_paraphrasing=False,  # Set to True to use LLM paraphrasing
        output_file="paraphrase_results.json"
    )
    """
    print("This is a library module. Import and use evaluate_wikidata_with_paraphrases()")
