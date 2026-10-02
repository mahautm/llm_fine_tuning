#!/usr/bin/env python3
"""
Standalone script to run paraphrase-based stability evaluation on a trained checkpoint.

Usage:
    srun --ntasks=1 python3 scripts/analysis/run_paraphrase_eval.py \\
        --checkpoint models2/Llama-3.1-8B-Instruct-wikiplus-full-ft/checkpoint-42 \\
        --dataset data3/wikidata_Mis7.csv \\
        --output paraphrase_results.json \\
        --num-samples 200 \\
        --num-paraphrases 5 \\
        --use-llm-paraphrasing
"""

import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from paramem.evaluation.wikidata_paraphrase_eval import evaluate_wikidata_with_paraphrases


def main():
    parser = argparse.ArgumentParser(description="Run paraphrase-based stability evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="data3/wikidata_Mis7.csv", 
                       help="Path to wikidata CSV file")
    parser.add_argument("--output", type=str, default="paraphrase_results.json",
                       help="Output JSON file path")
    parser.add_argument("--num-samples", type=int, default=200,
                       help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--num-paraphrases", type=int, default=5,
                       help="Number of paraphrases per question (default: 5)")
    parser.add_argument("--use-llm-paraphrasing", action="store_true",
                       help="Use LLM-based paraphrasing (slower but better)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save detailed results to JSON")
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)
    
    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Device: {device}")
    
    # Run evaluation
    print(f"\nRunning paraphrase evaluation:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Paraphrases per question: {args.num_paraphrases}")
    print(f"  Paraphrasing method: {'LLM-based' if args.use_llm_paraphrasing else 'rule-based'}")
    print(f"  Output: {args.output if not args.no_save else 'None (not saving)'}")
    print()
    
    output_file = None if args.no_save else args.output
    
    metrics = evaluate_wikidata_with_paraphrases(
        csv_path=args.dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_samples=args.num_samples,
        num_paraphrases=args.num_paraphrases,
        use_llm_paraphrasing=args.use_llm_paraphrasing,
        output_file=output_file
    )
    
    # Print results
    print("\n" + "="*60)
    print("PARAPHRASE STABILITY EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy:              {metrics['wikidata_paraphrase_overall_accuracy']:.4f}")
    print(f"Consistency Rate:              {metrics['wikidata_paraphrase_consistency_rate']:.4f}")
    print(f"Accuracy (when consistent):    {metrics['wikidata_paraphrase_acc_when_consistent']:.4f}")
    print(f"Accuracy (when inconsistent):  {metrics['wikidata_paraphrase_acc_when_inconsistent']:.4f}")
    print(f"Num Consistent Questions:      {metrics['wikidata_paraphrase_num_consistent']}")
    print(f"Total Questions:               {metrics['wikidata_paraphrase_num_questions']}")
    print("="*60)
    
    if output_file and os.path.exists(output_file):
        print(f"\nDetailed results saved to: {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
