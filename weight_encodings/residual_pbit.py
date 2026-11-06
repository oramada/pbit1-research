from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualPBitWeights:
    """A coarse p-bit plus optional smaller-scale residual p-bit."""

    p_coarse: np.ndarray
    p_residual: np.ndarray
    center: float
    radius: float
    residual_scale: float
    use_residual: np.ndarray


class ResidualPBit:
    """Hierarchical stochastic scalar code.

    The first p-bit uses a coarse probability selected on a grid with step
    ``2 * residual_scale * radius``. A second p-bit encodes the bounded
    difference exactly in expectation. The residual bit may be disabled per
    weight, yielding an adaptive spatial count. The expected decoded weight is
    unchanged; the purpose is to test the variance/bit-count tradeoff.
    """

    def __init__(self, residual_scale: float = 0.25, eps: float = 1e-6):
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if not 0.0 <= eps < 0.5:
            raise ValueError("eps must be in [0, 0.5)")
        self.residual_scale = float(residual_scale)
        self.eps = eps

    def encode(self, weights: np.ndarray, use_residual: np.ndarray | None = None,
               center: float | None = None, radius: float | None = None) -> ResidualPBitWeights:
        w = np.asarray(weights, dtype=np.float64)
        if center is None:
            low, high = float(np.min(w)), float(np.max(w))
            center = (low + high) / 2.0
        if radius is None:
            low, high = float(np.min(w)), float(np.max(w))
            radius = (high - low) / 2.0
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        if use_residual is None:
            use_residual = np.ones(w.shape, dtype=bool)
        else:
            use_residual = np.broadcast_to(np.asarray(use_residual, dtype=bool), w.shape).copy()
        if radius == 0:
            return ResidualPBitWeights(np.full(w.shape, 0.5), np.full(w.shape, 0.5),
                                      float(center), 0.0, self.residual_scale, use_residual)

        x = np.clip(w - center, -radius, radius)
        delta = self.residual_scale * radius
        # Quantized coarse expectation, constrained so the residual fits in +/-delta.
        coarse = np.rint(x / (2.0 * delta)) * (2.0 * delta)
        coarse = np.clip(coarse, -radius, radius)
        residual = x - coarse
        # Floating-point roundoff at endpoints can exceed delta by a few ulps.
        residual = np.clip(residual, -delta, delta)
        p_coarse = 0.5 * (coarse / radius + 1.0)
        p_residual = 0.5 * (residual / delta + 1.0)
        p_coarse = np.clip(p_coarse, self.eps, 1.0 - self.eps)
        p_residual = np.clip(p_residual, self.eps, 1.0 - self.eps)
        # If residual correction is disabled, preserve unbiasedness with the
        # one-p-bit probability mapping rather than the coarse approximation.
        p_one = 0.5 * (x / radius + 1.0)
        p_coarse = np.where(use_residual, p_coarse, np.clip(p_one, self.eps, 1-self.eps))
        return ResidualPBitWeights(p_coarse, p_residual, float(center), float(radius),
                                  self.residual_scale, use_residual)

    def sample(self, encoded: ResidualPBitWeights, n_samples: int,
               rng: np.random.Generator) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        coarse_spins = 2 * rng.binomial(1, encoded.p_coarse,
                                        size=(n_samples, *encoded.p_coarse.shape)) - 1
        residual_spins = 2 * rng.binomial(1, encoded.p_residual,
                                          size=(n_samples, *encoded.p_residual.shape)) - 1
        values = encoded.center + encoded.radius * coarse_spins
        residual_values = encoded.radius * encoded.residual_scale * residual_spins
        return values + residual_values * encoded.use_residual[None, ...]

    def decode(self, samples: np.ndarray) -> np.ndarray:
        return np.mean(samples, axis=0)

    def expected(self, encoded: ResidualPBitWeights) -> np.ndarray:
        coarse = encoded.center + encoded.radius * (2 * encoded.p_coarse - 1)
        residual = encoded.radius * encoded.residual_scale * (2 * encoded.p_residual - 1)
        return coarse + residual * encoded.use_residual

    def n_pbits(self, encoded: ResidualPBitWeights) -> int:
        return int(encoded.p_coarse.size + np.count_nonzero(encoded.use_residual))
