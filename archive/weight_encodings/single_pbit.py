from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SinglePBitWeights:
    probabilities: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    group_ids: np.ndarray
    n_groups: int


class SinglePBit:
    """One Bernoulli state per weight with affine local-range decoding."""

    def __init__(self, eps: float = 1e-6):
        if not 0.0 <= eps < 0.5:
            raise ValueError("eps must be in [0, 0.5)")
        self.eps = eps

    @staticmethod
    def _groups(shape: tuple[int, ...], group_ids: np.ndarray | None) -> np.ndarray:
        if group_ids is None:
            return np.zeros(shape, dtype=np.int64)
        ids = np.asarray(group_ids, dtype=np.int64)
        if ids.shape != shape:
            raise ValueError("group_ids must have the same shape as weights")
        _, inverse = np.unique(ids, return_inverse=True)
        return inverse.reshape(shape)

    def encode(self, weights: np.ndarray, group_ids: np.ndarray | None = None) -> SinglePBitWeights:
        w = np.asarray(weights, dtype=np.float64)
        ids = self._groups(w.shape, group_ids)
        flat_w, flat_ids = w.ravel(), ids.ravel()
        n_groups = int(flat_ids.max()) + 1 if flat_ids.size else 0
        centers = np.zeros(n_groups, dtype=np.float64)
        radii = np.zeros(n_groups, dtype=np.float64)
        probabilities = np.zeros_like(flat_w)
        for group in range(n_groups):
            mask = flat_ids == group
            low, high = float(np.min(flat_w[mask])), float(np.max(flat_w[mask]))
            centers[group] = (low + high) / 2.0
            radii[group] = (high - low) / 2.0
            if radii[group] == 0.0:
                probabilities[mask] = 0.5
            else:
                probabilities[mask] = 0.5 * ((flat_w[mask] - centers[group]) / radii[group] + 1.0)
        probabilities = np.clip(probabilities, self.eps, 1.0 - self.eps)
        return SinglePBitWeights(probabilities.reshape(w.shape), centers, radii,
                                 ids, n_groups)

    def sample(self, encoded_weights: SinglePBitWeights, n_samples: int,
               rng: np.random.Generator) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        bits = rng.binomial(1, encoded_weights.probabilities,
                            size=(n_samples, *encoded_weights.probabilities.shape))
        spins = 2.0 * bits - 1.0
        c = encoded_weights.centers[encoded_weights.group_ids]
        r = encoded_weights.radii[encoded_weights.group_ids]
        return c[None, ...] + r[None, ...] * spins

    def decode(self, samples: np.ndarray) -> np.ndarray:
        return np.mean(samples, axis=0)

    def forward(self, inputs: np.ndarray, encoded_weights: SinglePBitWeights,
                n_samples: int, rng: np.random.Generator) -> np.ndarray:
        samples = self.sample(encoded_weights, n_samples, rng)
        outputs = np.einsum("...i,tio->...to", np.asarray(inputs), samples)
        return np.mean(outputs, axis=-2)
