#!/usr/bin/env python3
"""
Create a MMLU training dataset in .txt format similar to pile_19_short.txt
"""

from datasets import load_dataset
from pathlib import Path
import random

def format_mmlu_qa(item):
    """Format a single MMLU item as question + answer text."""
    question = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    
    # Format as: Question with choices, then answer
    text = f"Question: {question}\n"
    text += f"A) {choices[0]}\n"
    text += f"B) {choices[1]}\n"
    text += f"C) {choices[2]}\n"
    text += f"D) {choices[3]}\n"
    text += f"Answer: {chr(65 + answer_idx)}) {choices[answer_idx]}\n"
    
    return text


def create_mmlu_dataset(output_path: str, num_samples: int = 6170, split: str = "auxiliary_train"):
    """
    Create MMLU training dataset matching the size of wikiplus/pile datasets.
    
    Args:
        output_path: Path to save the .txt file
        num_samples: Number of samples to include (default 6170 to match other datasets)
        split: Which MMLU split to use (auxiliary_train, dev, validation, test)
    """
    print(f"📥 Loading MMLU dataset from Hugging Face (split={split})...")
    
    # MMLU has multiple subjects, load the "all" configuration
    dataset = load_dataset("cais/mmlu", "all", split=split)
    
    print(f"📊 Original dataset size: {len(dataset)} samples")
    
    # If we need more samples than available, we'll repeat some
    if len(dataset) < num_samples:
        print(f"⚠️  Dataset has fewer samples than requested. Will repeat to reach {num_samples}.")
        # Repeat the dataset to get enough samples
        repetitions = (num_samples // len(dataset)) + 1
        indices = list(range(len(dataset))) * repetitions
        random.shuffle(indices)
        indices = indices[:num_samples]
    else:
        # Randomly sample if we have more than needed
        indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"📝 Generating {num_samples} formatted Q&A samples...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in indices:
            item = dataset[int(idx)]
            formatted_text = format_mmlu_qa(item)
            f.write(formatted_text + "\n")
    
    print(f"✅ Created MMLU dataset: {output_file}")
    print(f"   Total samples: {num_samples}")
    
    # Verify file size
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"   Total lines: {len(lines)}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create MMLU training dataset")
    parser.add_argument("--output", "-o", 
                       default="/home/mmahaut/projects/paramem/data/mmlu_train.txt",
                       help="Output file path")
    parser.add_argument("--num-samples", "-n", type=int, default=6170,
                       help="Number of samples (default: 6170 to match wikiplus/pile)")
    parser.add_argument("--split", "-s", default="auxiliary_train",
                       choices=["auxiliary_train", "dev", "validation", "test"],
                       help="MMLU split to use")
    
    args = parser.parse_args()
    
    create_mmlu_dataset(args.output, args.num_samples, args.split)
