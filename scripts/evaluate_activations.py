import torch
import argparse
import os
from src.sae.model import SparseAutoencoder
from src.sae.topk_model import TopKAutoencoder, BatchTopKAutoencoder
from src.sae.mp_utils import compute_dataset_mp_threshold, identify_domain_features

def run_eval(sae_path: str, acts_path: str, output_path: str):
    """Passes dataset activations through a pretrained SAE and saves the sparse feature activations."""
    print(f"Loading SAE from {sae_path}")
    checkpoint = torch.load(sae_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_type = checkpoint.get('sae_type', 'standard')
    
    if sae_type == 'standard':
        sae = SparseAutoencoder(
            d_model=checkpoint['d_model'], 
            dict_size=checkpoint['dict_size'], 
            l1_coeff=checkpoint['l1_coeff']
        ).to(device)
    elif sae_type == 'topk':
        sae = TopKAutoencoder(
            d_model=checkpoint['d_model'], 
            dict_size=checkpoint['dict_size'], 
            k=checkpoint['top_k']
        ).to(device)
    elif sae_type == 'batch_topk':
        sae = BatchTopKAutoencoder(
            d_model=checkpoint['d_model'], 
            dict_size=checkpoint['dict_size'], 
            # We don't save batch size natively in the model state, but batch_k is fixed per batch size 
            # This evaluation runs in chunks, so k_per_batch depends on chunk size.
            k_per_batch=checkpoint['top_k'] * 8192
        ).to(device)
        
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    print(f"Loading activations from {acts_path}")
    acts = torch.load(acts_path)
    if not isinstance(acts, torch.Tensor):
        acts = torch.tensor(acts)
        
    acts = acts.to(device).float()
    
    # Process in chunks to avoid GPU memory explosion if dataset is huge
    chunk_size = 8192
    f_acts_list = []
    
    print(f"Evaluating SAE features...")
    with torch.no_grad():
        for i in range(0, acts.shape[0], chunk_size):
            x_chunk = acts[i:i+chunk_size]
            # only want the sparse feature activations (f)
            _, _, f_chunk, _, _ = sae(x_chunk)
            f_acts_list.append(f_chunk.cpu())
            
    f_all = torch.cat(f_acts_list, dim=0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(f_all, output_path)
    print(f"Saved sparse feature activations (shape {f_all.shape}) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_model", type=str, required=True, help="Path to trained SAE .pt")
    parser.add_argument("--acts_in", type=str, required=True, help="Path to input dense activations .pt")
    parser.add_argument("--acts_out", type=str, required=True, help="Path to save sparse feature activations")
    
    args = parser.parse_args()
    run_eval(args.sae_model, args.acts_in, args.acts_out)
