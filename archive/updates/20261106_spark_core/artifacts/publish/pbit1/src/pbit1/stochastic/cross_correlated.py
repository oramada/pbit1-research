"""Cross-device correlation model for p-bit arrays.

The sampler uses a mixture copula: each simultaneous draw is independent with
probability 1-lambda, and uses one shared uniform latent variate with probability
lambda. Thresholding the shared variate at each p_i preserves each Bernoulli
marginal exactly. For equal probabilities this gives pairwise correlation lambda;
for unequal probabilities the Pearson correlation is pair-dependent, so lambda is
reported as the common-mode mixture fraction rather than an exact universal rho.
"""
from __future__ import annotations

import numpy as np


def sample_cross_correlated(probabilities: np.ndarray, n_samples: int,
                            common_mode: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Return Bernoulli draws shaped ``(n_samples, *probabilities.shape)``."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if not 0.0 <= common_mode <= 1.0:
        raise ValueError("common_mode must be in [0, 1]")
    p = np.asarray(probabilities, dtype=np.float64)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    shape = (n_samples, *p.shape)
    independent = rng.random(shape) < p
    shared_u = rng.random((n_samples,) + (1,) * p.ndim)
    common = shared_u < p
    use_common = rng.random((n_samples,) + (1,) * p.ndim) < common_mode
    return np.where(use_common, common, independent).astype(np.uint8)


def theoretical_dot_variance(probabilities: np.ndarray, coefficients: np.ndarray,
                             common_mode: float) -> float:
    """Exact one-draw variance for the mixture sampler.

    Computes covariance under the mixture of independent Bernoulli variables and
    thresholded shared-uniform Bernoulli variables.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    x = np.asarray(coefficients, dtype=np.float64).ravel()
    if p.shape != x.shape:
        raise ValueError("probabilities and coefficients must have same shape")
    lam = common_mode
    variances = p * (1.0 - p)
    covariance = lam * (np.minimum(p[:, None], p[None, :]) - p[:, None] * p[None, :])
    np.fill_diagonal(covariance, variances)
    return float(4.0 * (x @ covariance @ x))
