import sys
import pickle
import numpy as np
from dadapy import Data
import glob
import re
import concurrent
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def matrix_based_entropy(
    Z: torch.Tensor,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Compute the matrix-based entropy of the given data. we use the formulation from https://arxiv.org/abs/2502.02013
    Z has N samples and D dimensions, the entropy is computed as follows:
    Equation 1 in the paper:
    S_alpha(Z) = 1 / (alpha - 1) * log(sum_{i=1}^r lambda_i(K) / tr(K))^alpha)

    where K is the Gram matrix K = Z^T Z,
    r is the rank of K with r <= min(N, D), 
    lambda_i(K) are the eigenvalues of K, 
    and tr(K) is the trace of K.    
    
    Parameters:
        Z (np.ndarray): The input data matrix.
        
    Returns:
        np.ndarray: The computed entropy values.
    """
    if not isinstance(Z, torch.Tensor):
        raise TypeError("Input Z must be a torch.Tensor")
    
    if Z.ndim != 2:
        raise ValueError("Input Z must be a 2D tensor")
    
    # Compute the Gram matrix
    K = torch.mm(Z.T, Z)
    
    # Compute the eigenvalues and trace
    eigenvalues = torch.linalg.eigvalsh(K)
    trace_K = torch.trace(K)
    
    # Filter out zero eigenvalues
    non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
    r = non_zero_eigenvalues.shape[0]
    if r == 0:
        return np.array([0.0])

    # Compute the entropy
    if alpha == 1.0:
        # we cannot divide by zero, so we use the limit case
        entropy = torch.log(torch.sum(non_zero_eigenvalues) / trace_K)
    else:
        entropy = (1 / (alpha - 1)) * torch.log(
            torch.sum(non_zero_eigenvalues / trace_K) ** alpha)
    
    return entropy
    
    

if __name__ == "__main__":
    pickle_path = sys.argv[1]
    model_name = sys.argv[2]
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    with open(pickle_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    ents = {}
    for i, (k,v) in enumerate(data.items()):
        print(f"Block {k} has shape {np.array(v).shape}")
        ent=matrix_based_entropy(torch.tensor(v, dtype=torch.float32), alpha=alpha)
        print(f"Entropy for block {k}: {ent.item()}")
        ents[k] = ent.item()

    df = pd.DataFrame(list(ents.items()), columns=['Block', 'Entropy'])
    save_path = f"{model_name}_matrix_entropy.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Block', y='Entropy')
    plt.title(f'Matrix-Based Entropy per Block, alpha={alpha}')
    plt.xlabel('Block')
    plt.ylabel('Entropy')
    plt.savefig(save_path)
    plt.close()

