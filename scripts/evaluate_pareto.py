import torch
import argparse
import json
import os
import glob
import torch.nn.functional as F
from src.sae.mp_utils import compute_dataset_mp_threshold

def evaluate_pareto(models_dir: str, eval_acts_path: str, output_csv: str):
    """
    Evaluates all trained SAE models in a directory on a test set of activations 
    to map the Pareto frontier between Reconstruction (MSE) and Sparsity (L0/Active Nodes).
    It also checks the 'MP Pruned' reconstruction, showing if deleting features below
    MP noise preserves MSE.
    """
    from scripts.causal_intervention import load_sae_from_checkpoint
    
    print(f"Loading test activations from {eval_acts_path}...")
    acts = torch.load(eval_acts_path)
    if not isinstance(acts, torch.Tensor):
        acts = torch.tensor(acts)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acts = acts.to(device).float()
    
    # We sample a smaller chunk to avoid OOM during eval
    if acts.shape[0] > 10000:
        idx = torch.randperm(acts.shape[0])[:10000]
        acts = acts[idx]
        
    print(f"Test batch shape: {acts.shape}")
    
    results = []
    
    # Find all trained SAEs
    sae_files = glob.glob(os.path.join(models_dir, "*.pt"))
    token = os.path.basename(eval_acts_path).replace("_acts.pt", "")
    filtered = [sf for sf in sae_files if token in os.path.basename(sf)]
    if filtered:
        sae_files = filtered
        print(f"Found {len(sae_files)} matching SAE checkpoints to evaluate (token='{token}').")
    else:
        print(f"Found {len(sae_files)} SAE checkpoints to evaluate.")
    
    for sf in sae_files:
        try:
            print(f"Evaluating {os.path.basename(sf)}...")
            sae = load_sae_from_checkpoint(sf, device=device)
            
            with torch.no_grad():
                # 1. Base inference
                f_full = sae.encode(acts)
                x_recon_full = sae.decode(f_full)
                
                mse_base = F.mse_loss(x_recon_full, acts).item()
                # L0 Sparsity (average non-zero features per token)
                l0_base = (f_full > 1e-5).float().sum(dim=-1).mean().item()
                
                # 2. MP-Pruned Inference
                # Compute threshold over this dataset eval
                mp_mask, lmax, _, feature_vars = compute_dataset_mp_threshold(f_full)
                
                # Zero out features inside the "noise" bulk
                f_pruned = f_full.clone()
                f_pruned[:, ~mp_mask] = 0.0
                
                x_recon_pruned = sae.decode(f_pruned)
                mse_pruned = F.mse_loss(x_recon_pruned, acts).item()
                l0_pruned = (f_pruned > 1e-5).float().sum(dim=-1).mean().item()
                
                checkpoint = torch.load(sf, map_location="cpu")
                
                results.append({
                    "Model": os.path.basename(sf),
                    "Type": checkpoint.get("sae_type", "standard"),
                    "Param_Info": checkpoint.get("top_k", checkpoint.get("l1_coeff", "Unknown")),
                    "L0_Sparsity_Base": round(l0_base, 2),
                    "MSE_Base": round(mse_base, 6),
                    "L0_Sparsity_Pruned": round(l0_pruned, 2),
                    "MSE_Pruned": round(mse_pruned, 6),
                    "Features_Dropped_By_MP": int((~mp_mask).sum().item())
                })
                
        except Exception as e:
            print(f"Error evaluating {sf}: {e}")
            
    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    import csv
    if len(results) > 0:
        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    
        print(f"\nPareto front results written to {output_csv}")
    else:
        print("No models evaluated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing .pt SAE models")
    parser.add_argument("--test_acts", type=str, required=True, help="Test set activations .pt file")
    parser.add_argument("--out_csv", type=str, default="results/pareto_front.csv", help="Output file path")
    
    args = parser.parse_args()
    evaluate_pareto(args.models_dir, args.test_acts, args.out_csv)
