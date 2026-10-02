import numpy as np

from paramem.spectral.activation_cov import (
    compute_cov,
    compute_eigenspectrum,
    project_onto_singular_basis,
    stream_covariance,
)
from paramem.spectral.mp_fit import fit_mp, fit_mp_mle, partition


def _sample_wishart_eigenvalues(n: int, d: int, sigma2: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=(n, d))
    cov = (x.T @ x) / float(n)
    return np.linalg.eigvalsh(cov)


def test_fit_mp_recovery_on_wishart():
    n, d = 5000, 400
    gamma_eff = min(n, d) / max(n, d)
    sigma2_true = 2.0

    evals = _sample_wishart_eigenvalues(n=n, d=d, sigma2=sigma2_true, seed=42)
    fit = fit_mp(evals, gamma=gamma_eff)

    assert np.isclose(fit.sigma2, sigma2_true, atol=0.12)


def test_fit_mp_mle_returns_valid_edges():
    n, d = 3000, 300
    evals = _sample_wishart_eigenvalues(n=n, d=d, sigma2=1.0, seed=7)
    fit = fit_mp_mle(evals)

    assert 0.0 < fit.gamma <= 1.0
    assert fit.lambda_min >= 0.0
    assert fit.lambda_max > fit.lambda_min


def test_partition_detects_spike_direction():
    n, d = 5000, 200
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n, d))
    x[:, 0] *= 4.0

    cov = compute_cov(x, center=True)
    evals = np.linalg.eigvalsh(cov)

    gamma_eff = min(n, d) / max(n, d)
    fit = fit_mp(evals, gamma=gamma_eff)
    signal_mask, _ = partition(evals, fit.lambda_max)

    assert signal_mask.sum() >= 1


def test_stream_covariance_matches_batch_covariance():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(2048, 32))

    cov_batch = compute_cov(x)
    chunks = [x[:700], x[700:1400], x[1400:]]
    cov_stream = stream_covariance(chunks)

    assert np.allclose(cov_batch, cov_stream, atol=1e-8)


def test_project_onto_basis_diagonal_nonnegative_for_identity_basis():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(512, 16))
    cov = compute_cov(x)
    basis = np.eye(16)

    diag = project_onto_singular_basis(cov, basis)
    assert diag.shape == (16,)
    assert np.all(diag >= -1e-10)


def test_compute_eigenspectrum_descending_order():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(600, 20))
    cov = compute_cov(x)
    evals = compute_eigenspectrum(cov, descending=True)

    assert np.all(evals[:-1] >= evals[1:])
