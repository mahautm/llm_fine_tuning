import torch
import torch.optim as optim
import argparse
import os
from torch.utils.data import TensorDataset, DataLoader
from src.sae.model import SparseAutoencoder
from src.sae.topk_model import TopKAutoencoder, BatchTopKAutoencoder
from tqdm import tqdm

def train_sae(activations_path: str, expansion_factor: int, l1_coeff: float, batch_size: int, epochs: int, output_dir: str, sae_type: str = "standard", top_k: int = 32):
    print(f"Loading activations from {activations_path}...")
    activations = torch.load(activations_path)
    
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = activations.shape[-1]
    dict_size = d_model * expansion_factor
    
    print(f"Training {sae_type.upper()} SAE (d_model={d_model}, dict_size={dict_size}) on device={device}")
    
    if sae_type == "standard":
        sae = SparseAutoencoder(d_model=d_model, dict_size=dict_size, l1_coeff=l1_coeff).to(device)
    elif sae_type == "topk":
        sae = TopKAutoencoder(d_model=d_model, dict_size=dict_size, k=top_k).to(device)
    elif sae_type == "batch_topk":
        k_per_batch = top_k * batch_size # Equivalent sparsity density to token-wise topk
        sae = BatchTopKAutoencoder(d_model=d_model, dict_size=dict_size, k_per_batch=k_per_batch).to(device)
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")
        
    optimizer = optim.AdamW(sae.parameters(), lr=1e-3)
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    sae.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch[0].to(device)
            if x.dtype != torch.float32:
                x = x.float()
                
            optimizer.zero_grad()
            
            # Forward pass returns: loss, x_reconstructed, f, mse_loss, l1_loss
            loss, _, _, mse_loss, l1_loss = sae(x)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "MSE": f"{mse_loss.item():.4f}", "L1": f"{l1_loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_l1 = epoch_l1 / len(dataloader)
        print(f"Epoch {epoch+1} Completed - Avg Loss: {avg_loss:.4f} | Avg MSE: {avg_mse:.4f} | Avg L1: {avg_l1:.4f}")
        
    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(activations_path).replace("_acts.pt", "")
    save_path = os.path.join(output_dir, f"{sae_type}_sae_{basename}_ef{expansion_factor}.pt")
    
    torch.save({
        'model_state_dict': sae.state_dict(),
        'd_model': d_model,
        'dict_size': dict_size,
        'l1_coeff': l1_coeff,
        'sae_type': sae_type,
        'top_k': top_k
    }, save_path)
    
    print(f"SAE model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts_path", type=str, required=True, help="Path to the extracted activations .pt file")
    parser.add_argument("--expansion_factor", type=int, default=8, help="SAE dictionary expansion factor (d -> e*d)")
    parser.add_argument("--l1", type=float, default=1e-3, help="L1 regularization coefficient")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--out_dir", type=str, default="data/models/sae", help="Output directory to save SAE models")
    parser.add_argument("--type", type=str, default="standard", choices=["standard", "topk", "batch_topk"], help="Which SAE architecture to use.")
    parser.add_argument("--topk", type=int, default=32, help="K value for topk or batch_topk variants.")
    
    args = parser.parse_args()
    
    train_sae(args.acts_path, args.expansion_factor, args.l1, args.batch_size, args.epochs, args.out_dir, args.type, args.topk)
