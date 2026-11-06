from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DIMENSIONS = (16, 64, 256, 1024, 4096, 16384)
SAMPLE_COUNTS = (1, 4, 16, 64, 256)


def run(out_path: str = "results/dot_product_metrics.csv", seed: int = 7,
        n_trials: int = 32) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for dim in DIMENSIONS:
        w = rng.uniform(-1.0, 1.0, size=dim)
        x = rng.normal(size=dim) / np.sqrt(dim)
        p = (w + 1.0) / 2.0
        exact = float(w @ x)
        variance_one = float(np.sum(x * x * (1.0 - (2 * p - 1) ** 2)))
        for samples in SAMPLE_COUNTS:
            means = np.empty(n_trials, dtype=np.float64)
            chunk = max(1, min(64, n_trials, 8_000_000 // (samples * dim)))
            for begin in range(0, n_trials, chunk):
                count = min(chunk, n_trials - begin)
                bits = rng.binomial(1, p, size=(count, samples, dim))
                spins = 2.0 * bits - 1.0
                sampled_y = np.sum((spins * x[None, None, :]), axis=-1)
                means[begin:begin + count] = np.mean(sampled_y, axis=1)
            mse = float(np.mean((means - exact) ** 2))
            rows.append({"dimension": dim, "samples": samples, "exact_y": exact,
                         "empirical_mse": mse, "theoretical_mse": variance_one / samples,
                         "mse_ratio": mse / (variance_one / samples) if variance_one else 1.0,
                         "mean_abs_error": float(np.mean(np.abs(means - exact)))})
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    for result in run():
        print(result)
