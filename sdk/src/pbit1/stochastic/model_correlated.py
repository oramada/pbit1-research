from __future__ import annotations

import numpy as np


class MarkovPBitState:
    """Streaming binary Markov samples with a requested stationary Bernoulli p."""

    def __init__(self, probabilities: np.ndarray, rho: float,
                 rng: np.random.Generator):
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        self.p = np.asarray(probabilities, dtype=np.float64)
        self.rho = float(rho)
        self.rng = rng
        self.state = rng.binomial(1, self.p).astype(np.uint8)

    def next(self) -> np.ndarray:
        conditional_p = np.where(
            self.state == 1,
            self.p + self.rho * (1.0 - self.p),
            self.p * (1.0 - self.rho),
        )
        self.state = self.rng.binomial(1, conditional_p).astype(np.uint8)
        return self.state


def theoretical_effective_samples(n_samples: int, rho: float) -> float:
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    return n_samples * (1.0 - rho) / (1.0 + rho)
