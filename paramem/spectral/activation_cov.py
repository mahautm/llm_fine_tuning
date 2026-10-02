from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def compute_cov(
    activations: np.ndarray,
    center: bool = True,
    shrinkage: Optional[float] = None,
) -> np.ndarray:
    """
    Compute covariance of activations with shape [n_samples, n_features].
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("activations must be a 2D array [n_samples, n_features].")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")

    if center:
        x = x - x.mean(axis=0, keepdims=True)

    cov = (x.T @ x) / float(x.shape[0] - 1)

    if shrinkage is not None:
        alpha = float(shrinkage)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("shrinkage must be in [0, 1].")
        trace_mean = float(np.trace(cov) / cov.shape[0])
        cov = (1.0 - alpha) * cov + alpha * trace_mean * np.eye(cov.shape[0], dtype=cov.dtype)

    cov = 0.5 * (cov + cov.T)
    return cov


def stream_covariance(
    chunks: Iterable[np.ndarray],
    center: bool = True,
) -> np.ndarray:
    """
    Compute covariance from an iterable of chunks [chunk_n, n_features] without materializing all data.
    """
    n_total = 0
    mean = None
    m2 = None

    for chunk in chunks:
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("Each chunk must be 2D [chunk_n, n_features].")
        if x.shape[0] == 0:
            continue

        chunk_n = x.shape[0]
        chunk_mean = x.mean(axis=0)

        if n_total == 0:
            mean = chunk_mean.copy()
            xc = x - chunk_mean
            m2 = xc.T @ xc
            n_total = chunk_n
            continue

        assert mean is not None
        assert m2 is not None

        delta = chunk_mean - mean
        new_n = n_total + chunk_n

        xc = x - chunk_mean
        m2_chunk = xc.T @ xc

        mean = mean + delta * (chunk_n / new_n)
        m2 = m2 + m2_chunk + np.outer(delta, delta) * (n_total * chunk_n / new_n)
        n_total = new_n

    if n_total < 2:
        raise ValueError("Need at least 2 total samples across chunks.")

    if not center:
        raise ValueError("stream_covariance currently supports center=True only.")

    cov = m2 / float(n_total - 1)
    cov = 0.5 * (cov + cov.T)
    return cov


def compute_eigenspectrum(cov: np.ndarray, descending: bool = True) -> np.ndarray:
    """
    Return covariance eigenvalues.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square 2D matrix.")
    evals = np.linalg.eigvalsh(c)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    if descending:
        evals = evals[::-1]
    return evals


def compute_eigendecomposition(
    cov: np.ndarray,
    descending: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (eigenvalues, eigenvectors) for a covariance matrix.
    Eigenvectors are returned as columns.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square 2D matrix.")

    evals, evecs = np.linalg.eigh(c)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    if descending:
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
    return evals, evecs


def project_onto_singular_basis(
    cov: np.ndarray,
    right_singular_vectors: np.ndarray,
) -> np.ndarray:
    """
    Project covariance into basis V (columns of right_singular_vectors) and return diagonal energies.
    """
    c = np.asarray(cov, dtype=np.float64)
    v = np.asarray(right_singular_vectors, dtype=np.float64)

    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square 2D matrix.")
    if v.ndim != 2:
        raise ValueError("right_singular_vectors must be 2D.")
    if c.shape[0] != v.shape[0]:
        raise ValueError("Dimension mismatch between cov and basis matrix.")

    projected = v.T @ c @ v
    projected = 0.5 * (projected + projected.T)
    return np.diag(projected)


def project_activations(
    activations: np.ndarray,
    basis: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    """
    Project activation matrix [n_samples, n_features] into a column basis.
    """
    x = np.asarray(activations, dtype=np.float64)
    b = np.asarray(basis, dtype=np.float64)
    if x.ndim != 2 or b.ndim != 2:
        raise ValueError("activations and basis must be 2D arrays.")
    if x.shape[1] != b.shape[0]:
        raise ValueError("Dimension mismatch between activations and basis.")

    if center:
        x = x - x.mean(axis=0, keepdims=True)

    return x @ b
