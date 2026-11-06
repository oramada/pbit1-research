from __future__ import annotations

from typing import Protocol, Any

import numpy as np


class WeightEncoding(Protocol):
    """Common interface used by deterministic and probabilistic encodings."""

    def encode(self, weights: np.ndarray) -> Any: ...
    def sample(self, encoded_weights: Any, n_samples: int, rng: np.random.Generator) -> np.ndarray: ...
    def decode(self, samples: np.ndarray) -> np.ndarray: ...
    def forward(self, inputs: np.ndarray, encoded_weights: Any, n_samples: int,
                rng: np.random.Generator) -> np.ndarray: ...
