from __future__ import annotations

import numpy as np

from weight_encodings.fp_reference import FPReference
from weight_encodings.int4_reference import INT4Reference
from weight_encodings.pbit_multibit import MultiPBit
from weight_encodings.single_pbit import SinglePBit


def main() -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(-1, 1, size=(64, 32))
    x = rng.normal(size=(8, 64))
    encodings = [FPReference(), INT4Reference(), MultiPBit(2), MultiPBit(4),
                 MultiPBit(8), SinglePBit()]
    for encoding in encodings:
        code = encoding.encode(w)
        samples = encoding.sample(code, 64, rng)
        estimate = encoding.decode(samples)
        result = encoding.forward(x, code, 64, rng)
        print(f"{encoding.__class__.__name__:14s} estimate_rmse="
              f"{np.sqrt(np.mean((estimate - w) ** 2)):.6f} "
              f"forward_shape={result.shape}")


if __name__ == "__main__":
    main()
