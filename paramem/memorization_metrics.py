"""
Memorization metrics based on recent research:
- Discoverable memorization (https://arxiv.org/abs/2202.07646)
- Distributional memorization using n-gram correlation
- Intrinsic Dimension (ID) at embedding level

Minimal implementation to test memorization quantification during training.
"""

import torch
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


def _batched(iterable, batch_size):
    """Yield chunks from iterable with fixed batch_size."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def compute_discoverable_memorization(
    model,
    tokenizer, 
    training_sequences: List[str],
    context_length: int = 10,
    suffix_length: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute discoverable memorization: can the model complete training sequences
    when prompted with a prefix?
    
    Returns:
        - extractability_rate: fraction of sequences that can be extracted
        - avg_tokens_recovered: average number of correct tokens recovered
    """
    model.eval()
    correct_completions = 0
    total_tokens_correct = 0
    total_sequences = len(training_sequences)
    
    with torch.no_grad():
        for seq in training_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            
            if len(tokens) < context_length + suffix_length:
                continue
                
            # Use first context_length tokens as prompt
            prompt_tokens = tokens[:context_length]
            target_tokens = tokens[context_length:context_length + suffix_length]
            
            # Generate completion
            input_ids = torch.tensor([prompt_tokens]).to(device)
            output = model.generate(
                input_ids,
                max_new_tokens=suffix_length,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated = output[0][len(prompt_tokens):len(prompt_tokens) + suffix_length].cpu().tolist()
            
            # Check exact match
            if generated == target_tokens[:len(generated)]:
                correct_completions += 1
                
            # Count token-level matches
            for gen_tok, target_tok in zip(generated, target_tokens):
                if gen_tok == target_tok:
                    total_tokens_correct += 1
    
    return {
        "extractability_rate": correct_completions / total_sequences if total_sequences > 0 else 0,
        "avg_tokens_recovered": total_tokens_correct / (total_sequences * suffix_length) if total_sequences > 0 else 0
    }


def compute_ngram_overlap(token_ids: List[int], n: int = 3) -> Counter:
    """Compute n-gram frequencies for tokenized text."""
    ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    return Counter(ngrams)


def compute_distributional_memorization(
    model,
    tokenizer,
    training_sequences: List[str],
    n: int = 3,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute distributional memorization: correlation between model outputs
    and n-gram statistics from training data.
    
    Based on https://arxiv.org/pdf/2407.14985
    """
    model.eval()
    
    # Build n-gram model from training data using token IDs
    all_tokens = []
    for seq in training_sequences:
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    # We need (context length n) -> next token joint counts, so build n+1-grams
    ngram_counts = compute_ngram_overlap(all_tokens, n=n + 1)
    total_ngrams = sum(ngram_counts.values())
    
    # Compute correlation between model predictions and n-gram frequencies
    correlations = []
    
    with torch.no_grad():
        for seq in training_sequences[:min(100, len(training_sequences))]:  # Sample for efficiency
            tokens = tokenizer.encode(seq, add_special_tokens=False, return_tensors="pt").to(device)
            
            if tokens.shape[1] < n + 1:
                continue
                
            outputs = model(tokens, labels=tokens)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # For each position, compare model probability with n-gram probability
            for i in range(n, len(tokens[0]) - 1):
                # Get n-gram context
                context = tuple(tokens[0][i-n:i].cpu().tolist())
                next_token = tokens[0][i].item()
                
                # N-gram probability for this (context, next_token)
                ngram = context + (next_token,)
                ngram_prob = ngram_counts.get(ngram, 0) / total_ngrams if total_ngrams > 0 else 0
                
                # Model probability
                probs = torch.softmax(logits[i], dim=-1)
                model_prob = probs[next_token].cpu().item()
                
                if ngram_prob > 0:  # Only correlate where n-gram exists
                    correlations.append((model_prob, ngram_prob))
    
    if len(correlations) > 0:
        model_probs, ngram_probs = zip(*correlations)
        correlation = np.corrcoef(model_probs, ngram_probs)[0, 1]
    else:
        correlation = 0.0
    
    return {
        "ngram_correlation": correlation,
        "num_comparisons": len(correlations)
    }


def compute_train_nll_and_exposure(
    model,
    tokenizer,
    training_sequences: List[str],
    device: str = "cuda",
    max_length: int = 512,
    batch_size: int = 1,
    shuffle_baseline: bool = True
) -> Dict[str, float]:
    """Compute negative log-likelihood and a simple exposure-style delta.

    Exposure proxy: compare log-prob of the true sequence to a shuffled-token baseline.
    A large positive delta indicates the model assigns much higher likelihood to the
    exact training ordering than to a shuffled variant, signaling memorization.
    """
    model.eval()
    nlls: List[float] = []
    deltas: List[float] = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    rng = np.random.default_rng(0)

    with torch.no_grad():
        for batch in _batched(training_sequences, batch_size):
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            labels = enc.input_ids.clone()
            outputs = model(**enc, labels=labels)
            # loss is mean over tokens
            batch_loss = float(outputs.loss.detach().cpu().item())
            nlls.append(batch_loss)

            if shuffle_baseline:
                shuffled_inputs = []
                for ids in enc.input_ids:
                    ids_np = ids.cpu().numpy()
                    # Only shuffle the non-padding part, keep padding tail
                    non_pad_mask = ids_np != pad_id
                    real_len = int(non_pad_mask.sum())
                    if real_len == 0:
                        real_len = len(ids_np)
                    perm = rng.permutation(real_len)
                    shuffled_seq = np.concatenate([
                        ids_np[:real_len][perm],
                        np.full(len(ids_np) - real_len, pad_id, dtype=ids_np.dtype)
                    ])
                    shuffled_inputs.append(torch.tensor(shuffled_seq, dtype=ids.dtype))
                shuffled = torch.nn.utils.rnn.pad_sequence(
                    shuffled_inputs,
                    batch_first=True,
                    padding_value=pad_id
                ).to(device)
                shuffle_labels = shuffled.clone()
                shuffle_out = model(
                    input_ids=shuffled,
                    attention_mask=(shuffled != pad_id),
                    labels=shuffle_labels
                )
                shuffle_loss = float(shuffle_out.loss.detach().cpu().item())
                deltas.append(shuffle_loss - batch_loss)

    all_nll = np.array(nlls, dtype=np.float64) if len(nlls) else np.array([])
    all_delta = np.array(deltas, dtype=np.float64) if len(deltas) else np.array([])

    return {
        "train_nll": float(all_nll.mean()) if len(all_nll) else 0.0,
        "train_perplexity": float(np.exp(all_nll.mean())) if len(all_nll) else 0.0,
        "exposure_delta": float(all_delta.mean()) if len(all_delta) else 0.0,
        "exposure_positive_rate": float((all_delta > 0).mean()) if len(all_delta) else 0.0,
    }


def compute_embedding_intrinsic_dimension(
    embeddings: np.ndarray,
    k: int = 10
) -> float:
    """
    Estimate Intrinsic Dimension (ID) using TwoNN method.
    Higher ID = more complex/diverse representations = harder to memorize
    
    Based on https://aclanthology.org/2025.l2m2-1.2/
    
    Args:
        embeddings: [num_samples, embedding_dim]
        k: number of neighbors to consider
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_samples = embeddings.shape[0]
    if n_samples < k + 1:
        return 0.0
    
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    
    # TwoNN estimator: ID ≈ 1 / log(r2/r1)
    # where r1 and r2 are distances to 1st and 2nd nearest neighbors
    r1 = distances[:, 1]  # Skip self (index 0)
    r2 = distances[:, 2]
    
    # Avoid division by zero
    ratio = r2 / (r1 + 1e-10)
    log_ratio = np.log(ratio + 1e-10)
    
    # Average over all points
    id_estimate = np.mean(1.0 / (log_ratio + 1e-10))
    
    return float(id_estimate)


def get_sequence_embeddings(
    model,
    tokenizer,
    sequences: List[str],
    device: str = "cuda"
) -> np.ndarray:
    """Extract embeddings for sequences (mean pooling over tokens)."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for seq in sequences:
            tokens = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # Get embeddings from the model
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[0]  # First layer (embedding layer)
            
            # Convert to float32 for numpy compatibility
            hidden_states = hidden_states.float()
            
            # Mean pooling
            embedding = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(embedding[0])
    
    return np.array(embeddings)


def compute_all_memorization_metrics(
    checkpoint_path: str,
    training_sequences: List[str],
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all memorization metrics for a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        training_sequences: List of training sequences to test
        
    Returns:
        Dictionary with all metrics
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Computing metrics for {checkpoint_path}...")
    
    # Sample sequences for efficiency
    if max_samples and max_samples > 0:
        sample_seqs = np.random.choice(training_sequences, max_samples, replace=False).tolist()
    else:
        sample_seqs = list(training_sequences)
    
    # 1. Discoverable memorization
    print("  - Computing discoverable memorization...")
    disc_metrics = compute_discoverable_memorization(
        model, tokenizer, sample_seqs, device=device
    )
    
    # 2. Distributional memorization
    print("  - Computing distributional memorization...")
    dist_metrics = compute_distributional_memorization(
        model, tokenizer, sample_seqs, device=device
    )
    
    # 3. Intrinsic Dimension
    print("  - Computing intrinsic dimension...")
    embeddings = get_sequence_embeddings(model, tokenizer, sample_seqs[: min(50, len(sample_seqs))], device=device)
    id_estimate = compute_embedding_intrinsic_dimension(embeddings)

    # 4. Train NLL / exposure-style delta
    print("  - Computing train NLL / exposure...")
    nll_metrics = compute_train_nll_and_exposure(
        model,
        tokenizer,
        sample_seqs,
        device=device,
        batch_size=1
    )
    
    return {
        **disc_metrics,
        **dist_metrics,
        "intrinsic_dimension": id_estimate,
        **nll_metrics,
        "num_sequences_tested": len(sample_seqs)
    }


if __name__ == "__main__":
    import typer
    
    def main(
        checkpoint: str = typer.Argument(..., help="Path to model checkpoint"),
        data_file: str = typer.Option("data3/wikidata_Mis7.csv", help="Training data CSV"),
        output: str = typer.Option("memorization_metrics.json", help="Output JSON file"),
        num_samples: int = typer.Option(0, help="If >0, cap number of sequences; otherwise use all")
    ):
        """Compute memorization metrics for a checkpoint."""
        import json
        
        # Load training data
        df = pd.read_csv(data_file)
        if "query" in df.columns:
            seq_series = df["query"].dropna()
        elif "text" in df.columns:
            seq_series = df["text"].dropna()
        else:
            raise ValueError("Data file must have 'query' or 'text' column")

        if num_samples and num_samples > 0:
            seq_series = seq_series.head(num_samples)
        sequences = seq_series.tolist()
        
        # Compute metrics
        metrics = compute_all_memorization_metrics(checkpoint, sequences, max_samples=num_samples if num_samples > 0 else None)
        
        # Save results
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {output}")
        print(json.dumps(metrics, indent=2))
    
    typer.run(main)
