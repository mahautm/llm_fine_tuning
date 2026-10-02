import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader

def get_text_dataset(dataset_name: str, tokenizer, split="train", max_length=128, batch_size=16):
    """Loads a text dataset from HF Hub and tokenizes it."""
    # Special casing for local files vs hub datasets
    if dataset_name.endswith('.txt') or dataset_name.endswith('.csv'):
        # Usually requires more specialized loading, but keeping simple for example
        if dataset_name.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_name, split=split)
        else:
            dataset = load_dataset('text', data_files=dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
        
    # Text datasets usually have 'text' or 'content' keys
    text_column = 'text' if 'text' in dataset.column_names else 'content'
    if text_column not in dataset.column_names:
        text_column = dataset.column_names[0] # Fallback
        
    def tokenize_fn(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

def get_image_dataset(dataset_name: str, processor, split="train", batch_size=16):
    """Loads an image dataset from HF Hub and processes it."""
    dataset = load_dataset(dataset_name, split=split)
    
    # Image datasets usually have 'image' or 'img' keys
    img_column = 'image' if 'image' in dataset.column_names else 'img'
    
    def process_fn(examples):
        # Convert grayscale or RGBA to RGB just in case
        images = [img.convert('RGB') for img in examples[img_column]]
        return processor(images, return_tensors="pt")

    # Important: use set_transform for images rather than map to save memory/processing time
    dataset.set_transform(process_fn)
    
    # Return custom collate_fn for dictionary of tensors
    def collate_fn(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
