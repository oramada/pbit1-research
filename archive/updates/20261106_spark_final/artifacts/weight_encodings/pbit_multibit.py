from __future__ import annotations

import numpy as np


class MultiPBit:
    """K spatial bipolar bits represent a uniformly quantized value in [-1, 1].

    Bits are currently ideal and deterministic at inference. This is the spatial
    representation baseline; device nonidealities can be layered on in later phases.
    """

    def __init__(self, n_bits: int = 4):
        if n_bits not in (2, 4, 8):
            raise ValueError("n_bits must be one of 2, 4, or 8")
        self.n_bits = n_bits

    def encode(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(weights, dtype=np.float64)
        limit = (1 << self.n_bits) - 1
        # A K-spin weighted sum with coefficients 1,2,4,... spans every odd
        # quantization level from -(2^K-1) to +(2^K-1).
        levels = np.rint((np.clip(w, -1.0, 1.0) + 1.0) * limit / 2.0).astype(np.uint16)
        bit_positions = np.arange(self.n_bits, dtype=np.uint16)
        planes = ((levels[..., None] >> bit_positions) & 1).astype(np.int8)
        spins = np.where(planes == 1, 1, -1).astype(np.int8)
        coeffs = (2.0 ** bit_positions.astype(np.float64)) / limit
        return spins, coeffs

    def sample(self, encoded_weights: tuple[np.ndarray, np.ndarray], n_samples: int,
               rng: np.random.Generator) -> np.ndarray:
        spins, coeffs = encoded_weights
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        decoded = np.sum(spins * coeffs, axis=-1)
        return np.broadcast_to(decoded, (n_samples, *decoded.shape)).copy()

    def decode(self, samples: np.ndarray) -> np.ndarray:
        return np.mean(samples, axis=0)

    def forward(self, inputs: np.ndarray, encoded_weights: tuple[np.ndarray, np.ndarray],
                n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(inputs) @ self.decode(self.sample(encoded_weights, n_samples, rng))
