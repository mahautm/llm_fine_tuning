"""Utility functions for extracting layer representations on-the-fly from models."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from typing import Dict, List, Optional
from tqdm import tqdm
import logging


def extract_layer_representations(
    model_path: str,
    data_samples: List[str],
    batch_size: int = 16,
    device: Optional[str] = None,
    is_lora: bool = False,
    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
) -> Dict[int, List[np.ndarray]]:
    """
    Extract layer-wise representations from a model on given text samples.
    
    Args:
        model_path: Path to model checkpoint or HF model name
        data_samples: List of text strings to process
        batch_size: Batch size for processing
        device: Device to use (auto-detected if None)
        is_lora: Whether this is a LoRA checkpoint
        base_model_name: Base model name for LoRA loading
        
    Returns:
        Dictionary mapping layer indices to lists of representations
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"📥 Loading model from {model_path}")
    print(f"🔧 Device: {device}")
    
    # For Full-FT checkpoints, check if we need to use merged_single_model subdirectory
    actual_model_path = model_path
    if not is_lora:
        from pathlib import Path
        checkpoint_path = Path(model_path)
        merged_path = checkpoint_path / "merged_single_model"
        if merged_path.exists() and merged_path.is_dir():
            print(f"📁 Using merged model from: {merged_path}")
            actual_model_path = str(merged_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name if is_lora else actual_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA weights if needed
    if is_lora:
        print("🔄 Loading LoRA checkpoint...")
        # Use PeftModel.from_pretrained which handles both .bin and .safetensors
        from pathlib import Path
        adapter_path = Path(model_path)
        if adapter_path.is_file():
            # If pointing to a specific file, use parent directory
            adapter_path = adapter_path.parent
        
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"✅ Loaded LoRA adapter from {adapter_path}")
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name if is_lora else model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract representations
    states = {}
    num_batches = (len(data_samples) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting representations"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data_samples))
            batch_texts = data_samples[start_idx:end_idx]
            
            # Tokenize
            inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
            
            # Find last true token positions
            last_token_indices = []
            for att_mask in inputs.attention_mask:
                if 0 not in att_mask:
                    last_token_indices.append(len(att_mask) - 1)
                else:
                    last_token_indices.append(att_mask.tolist().index(0) - 1)
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            
            # Extract last token representations per layer
            for layer_idx, layer_hidden in enumerate(hidden_states):
                if layer_idx not in states:
                    states[layer_idx] = []
                
                for sample_idx, token_idx in enumerate(last_token_indices):
                    activation = layer_hidden[sample_idx, token_idx].cpu().numpy()
                    states[layer_idx].append(activation)
    
    print(f"✅ Extracted {len(states)} layers, {len(states[0])} samples")
    return states


def load_benchmark_data(
    benchmark_file: str,
    max_samples: Optional[int] = None
) -> tuple[List[str], List[str]]:
    """
    Load benchmark data from TSV or JSONL file.
    
    Args:
        benchmark_file: Path to TSV/JSONL file
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (texts, labels)
    """
    import json
    texts = []
    labels = []
    
    with open(benchmark_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Try JSONL first
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    # Try common fields
                    text = (data.get('template') or 
                           data.get('text') or 
                           data.get('query') or 
                           data.get('input') or 
                           data.get('prompt', ''))
                    
                    # For labels, try multiple fields
                    label = (data.get('result_qids') or
                            data.get('result_names') or
                            data.get('label') or 
                            data.get('answer') or 
                            data.get('target') or
                            data.get('property', ''))
                    
                    # Convert to string if needed
                    if isinstance(label, list) and label:
                        label = str(label[0])
                    
                    if text:
                        texts.append(str(text))
                        labels.append(str(label))
                except json.JSONDecodeError:
                    pass
            else:
                # Try TSV format
                parts = line.split('\t')
                if len(parts) >= 3:
                    labels.append(parts[1])
                    texts.append(parts[2])
                elif len(parts) == 2:
                    labels.append(parts[0])
                    texts.append(parts[1])
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    return texts, labels
