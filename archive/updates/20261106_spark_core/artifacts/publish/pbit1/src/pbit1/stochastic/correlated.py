from __future__ import annotations

import numpy as np


def sample_markov(probabilities: np.ndarray, n_samples: int, rho: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Stationary Bernoulli chains with lag-one correlation rho.

    For each element with stationary probability p, transitions are
    P(1|0)=p(1-rho) and P(0|1)=(1-p)(1-rho).
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    p = np.asarray(probabilities, dtype=np.float64)
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    out = np.empty((n_samples, *p.shape), dtype=np.uint8)
    out[0] = rng.binomial(1, p)
    for t in range(1, n_samples):
        previous = out[t - 1]
        conditional_p = np.where(previous == 1, p + rho * (1.0 - p), p * (1.0 - rho))
        out[t] = rng.binomial(1, conditional_p)
    return out


def calibrate_probability(probabilities: np.ndarray, sigma_a: float, sigma_b: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Apply multiplicative logit slope and additive logit offset errors."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    a = rng.normal(1.0, sigma_a, size=p.shape)
    b = rng.normal(0.0, sigma_b, size=p.shape)
    logits = np.log(p) - np.log1p(-p)
    calibrated_logits = a * logits + b
    return 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -40.0, 40.0)))


def effective_sample_size(n_samples: int, rho: float) -> float:
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    return n_samples * (1.0 - rho) / (1.0 + rho)
