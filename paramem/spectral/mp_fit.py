from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np


class MPFitResult(NamedTuple):
    sigma2: float
    gamma: float
    lambda_min: float
    lambda_max: float


def _as_sorted_eigenvalues(eigenvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    ev = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        raise ValueError("No finite eigenvalues provided.")
    ev = np.clip(ev, a_min=0.0, a_max=None)
    ev.sort()
    return ev


def _mp_edges(sigma2: float, gamma: float) -> tuple[float, float]:
    if sigma2 <= 0:
        raise ValueError("sigma2 must be > 0.")
    if not (0 < gamma <= 1):
        raise ValueError("gamma must be in (0, 1].")
    sqrt_g = np.sqrt(gamma)
    lambda_min = sigma2 * (1.0 - sqrt_g) ** 2
    lambda_max = sigma2 * (1.0 + sqrt_g) ** 2
    return float(lambda_min), float(lambda_max)


def fit_mp(eigenvalues: Sequence[float] | np.ndarray, gamma: Optional[float] = None) -> MPFitResult:
    """
    Moment-style MP fit.

    If gamma is unknown, infer it from first two moments:
    m1 = sigma2,
    m2 = sigma2^2 * (1 + gamma)  => gamma = m2/m1^2 - 1.
    """
    ev = _as_sorted_eigenvalues(eigenvalues)

    m1 = float(np.mean(ev))
    if m1 <= 0:
        raise ValueError("Mean eigenvalue is non-positive; cannot fit MP.")

    if gamma is None:
        m2 = float(np.mean(ev**2))
        gamma_hat = (m2 / (m1 * m1)) - 1.0
        gamma_hat = float(np.clip(gamma_hat, 1e-6, 1.0))
    else:
        gamma_hat = float(gamma)
        if not (0 < gamma_hat <= 1.0):
            raise ValueError("gamma must be in (0, 1].")

    sigma2_hat = m1
    lambda_min, lambda_max = _mp_edges(sigma2=sigma2_hat, gamma=gamma_hat)

    return MPFitResult(
        sigma2=float(sigma2_hat),
        gamma=float(gamma_hat),
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )


def fit_mp_mle(
    eigenvalues: Sequence[float] | np.ndarray,
    gamma_grid: Optional[Iterable[float]] = None,
) -> MPFitResult:
    """
    Lightweight pseudo-MLE: choose gamma minimizing edge mismatch with empirical quantiles.
    """
    ev = _as_sorted_eigenvalues(eigenvalues)

    if gamma_grid is None:
        gamma_grid = np.linspace(0.02, 1.0, 100)

    sigma2_hat = float(np.mean(ev))
    emp_low = float(np.quantile(ev, 0.01))
    emp_high = float(np.quantile(ev, 0.99))

    best_gamma = None
    best_loss = np.inf
    for candidate in gamma_grid:
        candidate = float(candidate)
        if not (0 < candidate <= 1.0):
            continue
        lambda_min, lambda_max = _mp_edges(sigma2_hat, candidate)
        loss = (lambda_min - emp_low) ** 2 + (lambda_max - emp_high) ** 2
        if loss < best_loss:
            best_loss = loss
            best_gamma = candidate

    if best_gamma is None:
        raise ValueError("No valid gamma candidates found for MP fit.")

    lambda_min, lambda_max = _mp_edges(sigma2=sigma2_hat, gamma=best_gamma)
    return MPFitResult(
        sigma2=sigma2_hat,
        gamma=best_gamma,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )


def partition(
    eigenvalues: Sequence[float] | np.ndarray,
    lambda_max: float,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split eigenvalues into signal/noise masks using MP upper bulk edge.
    """
    ev = _as_sorted_eigenvalues(eigenvalues)
    edge = float(lambda_max)
    signal_mask = ev > (edge + float(atol))
    noise_mask = ~signal_mask
    return signal_mask, noise_mask


def generate_mp_pdf_grid(
    fit: MPFitResult,
    num_points: int = 512,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lambda_grid, pdf_values) for diagnostics/plotting.
    """
    if num_points < 8:
        raise ValueError("num_points must be >= 8")

    lam = np.linspace(fit.lambda_min + eps, fit.lambda_max - eps, num_points)
    denom = 2.0 * np.pi * fit.sigma2 * fit.gamma * lam
    rad = (fit.lambda_max - lam) * (lam - fit.lambda_min)
    rad = np.clip(rad, 0.0, None)
    pdf = np.sqrt(rad) / np.maximum(denom, eps)
    return lam, pdf
