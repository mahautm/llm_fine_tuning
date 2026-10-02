import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Standard Sparse Autoencoder for extracting monosemantic features
    from model activations.
    """
    def __init__(self, d_model: int, dict_size: int, l1_coeff: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.l1_coeff = l1_coeff

        # Encoder: map model activations to higher-dimensional sparse features
        self.encoder = nn.Linear(d_model, dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        
        # Decoder: reconstruct original activations from sparse features
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Weight initialization (decoder weights normalized)
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        
    def encode(self, x):
        """Returns the sparse feature activations."""
        # x is assumed to be centered or pre-biased
        x_shifted = x - self.pre_bias
        return F.relu(self.encoder(x_shifted) + self.encoder_bias)

    def decode(self, f):
        """Reconstructs the original activation vector."""
        return self.decoder(f) + self.pre_bias

    def forward(self, x):
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        
        # Losses
        mse_loss = F.mse_loss(x_reconstructed, x)
        l1_loss = f.abs().sum(dim=-1).mean()
        
        loss = mse_loss + self.l1_coeff * l1_loss
        return loss, x_reconstructed, f, mse_loss, l1_loss
