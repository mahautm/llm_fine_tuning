import torch
import numpy as np
from scipy.optimize import brentq


def _robust_scale(
    activations: torch.Tensor,
    scale_mode: str,
    eps: float = 1e-8,
    min_scale: float = 1e-3,
) -> torch.Tensor:
    if scale_mode == "none":
        return activations

    if scale_mode == "zscore":
        std = activations.std(dim=0, keepdim=True)
        std = torch.where(std < min_scale, torch.ones_like(std), std).clamp_min(eps)
        return activations / std

    if scale_mode == "mad":
        med = activations.median(dim=0, keepdim=True).values
        mad = (activations - med).abs().median(dim=0, keepdim=True).values
        # 1.4826 rescales MAD to a Gaussian-consistent std estimate.
        denom = 1.4826 * mad
        denom = torch.where(denom < min_scale, torch.ones_like(denom), denom).clamp_min(eps)
        return activations / denom

    raise ValueError(f"Unknown scale_mode '{scale_mode}'.")


def _estimate_sigma_sq(
    eigenvalues: torch.Tensor,
    estimator: str,
    bottom_fraction: float,
    sigma_quantile: float,
) -> float:
    if estimator == "bottom_mean":
        k = max(1, int(bottom_fraction * len(eigenvalues)))
        return float(eigenvalues[:k].mean().item())

    if estimator == "median":
        return float(torch.median(eigenvalues).item())

    if estimator == "quantile":
        q = float(np.clip(sigma_quantile, 0.01, 0.99))
        return float(torch.quantile(eigenvalues, q).item())

    raise ValueError(f"Unknown sigma estimator '{estimator}'.")


def _estimate_abs_clip_value(
    values: torch.Tensor,
    quantile: float,
    max_samples: int = 2_000_000,
) -> torch.Tensor:
    flat = values.abs().reshape(-1)
    if flat.numel() > max_samples:
        # Deterministic stride subsampling to keep memory bounded on very large tensors.
        stride = max(1, flat.numel() // max_samples)
        flat = flat[::stride]
    return torch.quantile(flat, float(quantile))

def get_mp_lambda_max(N: int, P: int, sigma_sq: float = 1.0) -> float:
    """
    Computes the standard Marchenko-Pastur upper edge (bulk edge).
    
    Args:
        N: Number of samples (e.g., tokens)
        P: Number of dimensions (e.g., features)
        sigma_sq: Variance of the noise entries
        
    Returns:
        The theoretical maximum eigenvalue of pure noise.
    """
    gamma = P / N
    lambda_max = sigma_sq * (1 + np.sqrt(gamma))**2
    return lambda_max

def compute_dataset_mp_threshold(
    activations: torch.Tensor,
    estimate_noise_variance: bool = True,
    scale_mode: str = "none",
    clip_quantile: float | None = None,
    sigma_estimator: str = "bottom_mean",
    sigma_quantile: float = 0.5,
    bottom_fraction: float = 0.8,
) -> torch.Tensor:
    """
    Calculates the dataset-conditional Marchenko-Pastur threshold 
    for isolating True SAE features from the bulk.
    
    Args:
        activations: [num_tokens, num_features]
    
    Returns:
        A boolean mask [num_features] of which features exceed the MP noise floor.
    """
    N, P = activations.shape
    
    # Compute empirical covariance matrix using centered and optionally robust-scaled acts.
    centered_acts = activations - activations.mean(dim=0, keepdim=True)
    centered_acts = _robust_scale(centered_acts, scale_mode=scale_mode)

    if clip_quantile is not None:
        cq = float(np.clip(clip_quantile, 0.5, 0.9999))
        abs_q = _estimate_abs_clip_value(centered_acts, cq)
        centered_acts = centered_acts.clamp(min=-abs_q, max=abs_q)

    # 1/N * X^T X
    cov = (centered_acts.T @ centered_acts) / (N - 1)
    
    # Compute eigenvalues of covariance
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.nan_to_num(eigenvalues, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues = eigenvalues.clamp_min(0.0)
    
    if estimate_noise_variance:
        sigma_sq_est = _estimate_sigma_sq(
            eigenvalues=eigenvalues,
            estimator=sigma_estimator,
            bottom_fraction=float(np.clip(bottom_fraction, 0.05, 1.0)),
            sigma_quantile=sigma_quantile,
        )
    else:
        sigma_sq_est = 1.0
        
    l_max = get_mp_lambda_max(N, P, sigma_sq_est)
    
    # Any feature vector whose variance > l_max is considered "active" 
    # (a true feature, unmixed from the superposition substrate)
    feature_variances = torch.diag(cov)
    return feature_variances > l_max, l_max, sigma_sq_est, feature_variances

def identify_domain_features(general_mask: torch.Tensor, domain_mask: torch.Tensor):
    """
    Categorizes features into Universal, Domain, Dead based on dual-run thresholds.
    """
    universal = general_mask & domain_mask
    domain = (~general_mask) & domain_mask
    latent = (~general_mask) & (~domain_mask)  # If it fires later, currently dormant
    dead = latent # Simplified
    
    return {
        "universal": universal.nonzero().squeeze(),
        "domain": domain.nonzero().squeeze(),
        "dead": dead.nonzero().squeeze()
    }
