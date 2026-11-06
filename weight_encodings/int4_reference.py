from __future__ import annotations

import numpy as np


class INT4Reference:
    """Symmetric, per-tensor signed 4-bit quantization (levels -7 through 7)."""

    def encode(self, weights: np.ndarray) -> tuple[np.ndarray, float]:
        weights = np.asarray(weights, dtype=np.float64)
        scale = float(np.max(np.abs(weights)) / 7.0) if weights.size else 0.0
        if scale == 0.0:
            return np.zeros_like(weights, dtype=np.int8), 0.0
        q = np.clip(np.rint(weights / scale), -7, 7).astype(np.int8)
        return q, scale

    def sample(self, encoded_weights: tuple[np.ndarray, float], n_samples: int,
               rng: np.random.Generator) -> np.ndarray:
        q, scale = encoded_weights
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        return np.broadcast_to(q.astype(np.float64) * scale,
                               (n_samples, *q.shape)).copy()

    def decode(self, samples: np.ndarray) -> np.ndarray:
        return np.mean(samples, axis=0)

    def forward(self, inputs: np.ndarray, encoded_weights: tuple[np.ndarray, float],
                n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(inputs) @ self.decode(self.sample(encoded_weights, n_samples, rng))
