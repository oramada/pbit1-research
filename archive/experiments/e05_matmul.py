from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from weight_encodings.single_pbit import SinglePBit


def main() -> None:
    rng = np.random.default_rng(51)
    x = rng.normal(size=(32, 64))
    w = rng.normal(0.0, 0.08, size=(64, 48))
    state = SinglePBit().encode(w)
    expected = x @ w
    rows = []
    for n_samples in (1, 4, 16, 64, 256):
        sampled_weights = SinglePBit().sample(state, n_samples, rng)
        # Y^(t) = X W^(t), then average complete operator outputs.
        direct = np.mean(np.stack([x @ wt for wt in sampled_weights]), axis=0)
        reconstructed = x @ SinglePBit().decode(sampled_weights)
        rows.append({"samples": n_samples,
                     "direct_rmse": float(np.sqrt(np.mean((direct - expected) ** 2))),
                     "reconstructed_rmse": float(np.sqrt(np.mean((reconstructed - expected) ** 2))),
                     "two_modes_max_abs_difference": float(np.max(np.abs(direct - reconstructed)))})
    Path("results").mkdir(exist_ok=True)
    with open("results/matmul_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
