import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKAutoencoder(nn.Module):
    """
    Top-K Sparse Autoencoder. 
    Instead of using L1 regularization, it enforces exact sparsity k per token 
    by only keeping the top k activations and zeroing the rest.
    """
    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        # Encoder: map model activations to higher-dimensional features
        self.encoder = nn.Linear(d_model, dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        
        # Decoder: reconstruct original activations
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Weight initialization (decoder weights normalized)
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        
    def encode(self, x):
        """Returns the sparse feature activations."""
        x_shifted = x - self.pre_bias
        pre_acts = self.encoder(x_shifted) + self.encoder_bias
        
        # Apply Top-K
        # Efficiently get top k indices
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        
        # Create sparse representation
        sparse_acts = torch.zeros_like(pre_acts)
        # We need relu on topk values (though standard TopK SAEs sometimes don't use relu, 
        # usually they do to prevent negative features being "topk")
        sparse_acts.scatter_(-1, topk_indices, F.relu(topk_vals))
        
        return sparse_acts

    def decode(self, f):
        """Reconstructs the original activation vector."""
        return self.decoder(f) + self.pre_bias

    def forward(self, x):
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        
        # Loss: only MSE needed, sparsity is enforced natively by TopK
        mse_loss = F.mse_loss(x_reconstructed, x)
        
        # We keep the return signature compatible with standard SAE script
        # returning 0.0 for l1_loss
        return mse_loss, x_reconstructed, f, mse_loss, torch.tensor(0.0, device=x.device)


class BatchTopKAutoencoder(nn.Module):
    """
    Batch Top-K Sparse Autoencoder.
    Instead of K active features per token, this guarantees exact sparsity K 
    across an entire batch. Useful for balancing feature utilization globally.
    """
    def __init__(self, d_model: int, dict_size: int, k_per_batch: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k_per_batch = k_per_batch

        # Encoder
        self.encoder = nn.Linear(d_model, dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        
        # Decoder
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Initialize
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        
    def encode(self, x):
        x_shifted = x - self.pre_bias
        pre_acts = self.encoder(x_shifted) + self.encoder_bias
        
        # Flatten batch and feature dims to do global Top-K
        batch_size, num_features = pre_acts.shape
        flat_pre_acts = pre_acts.view(-1)
        
        # We handle case where batch is extremely small compared to k_per_batch
        actual_k = min(self.k_per_batch, flat_pre_acts.numel())
        
        # Global top k
        topk_vals, topk_indices = torch.topk(flat_pre_acts, actual_k)
        
        # Create sparse representation
        sparse_acts_flat = torch.zeros_like(flat_pre_acts)
        sparse_acts_flat.scatter_(0, topk_indices, F.relu(topk_vals))
        
        # Reshape back to [batch, dict_size]
        return sparse_acts_flat.view(batch_size, num_features)

    def decode(self, f):
        return self.decoder(f) + self.pre_bias

    def forward(self, x):
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        
        mse_loss = F.mse_loss(x_reconstructed, x)
        return mse_loss, x_reconstructed, f, mse_loss, torch.tensor(0.0, device=x.device)
