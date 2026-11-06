from __future__ import annotations

import numpy as np


class FPReference:
    def encode(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(weights, dtype=np.float64).copy()

    def sample(self, encoded_weights: np.ndarray, n_samples: int,
               rng: np.random.Generator) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        return np.broadcast_to(encoded_weights, (n_samples, *encoded_weights.shape)).copy()

    def decode(self, samples: np.ndarray) -> np.ndarray:
        return np.mean(samples, axis=0)

    def forward(self, inputs: np.ndarray, encoded_weights: np.ndarray, n_samples: int,
                rng: np.random.Generator) -> np.ndarray:
        return np.asarray(inputs) @ self.decode(self.sample(encoded_weights, n_samples, rng))
