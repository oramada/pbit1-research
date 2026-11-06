from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)


def run(n_targets: int = 1000, n_trials: int = 1000,
        out_path: str = "results/scalar_metrics.csv", seed: int = 20260924) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    targets = np.linspace(-1.0, 1.0, n_targets, dtype=np.float64)
    probabilities = ((targets + 1.0) / 2.0)[None, :]
    rows: list[dict[str, float]] = []
    for n_samples in SAMPLE_COUNTS:
        # Binomial sampling is exactly equivalent to drawing and averaging n_samples
        # independent Bernoulli values, while avoiding a 4-D random tensor.
        observations = rng.binomial(n_samples, probabilities, size=(n_trials, n_targets))
        estimates = 2.0 * observations / n_samples - 1.0
        errors = estimates - targets[None, :]
        empirical_var = float(np.mean(errors ** 2))
        theory_var = float(np.mean(4.0 * probabilities * (1.0 - probabilities)) / n_samples)
        rows.append({
            "samples": n_samples,
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(empirical_var)),
            "empirical_mse": empirical_var,
            "theoretical_mse": theory_var,
            "mse_ratio": empirical_var / theory_var if theory_var else 1.0,
            "worst_case_std": float(1.0 / np.sqrt(n_samples)),
            "effective_bits": float(0.5 * np.log2(n_samples)),
        })
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--out", default="results/scalar_metrics.csv")
    args = parser.parse_args()
    for row in run(args.targets, args.trials, args.out):
        print("T={samples:4d} bias={bias:+.3e} MAE={mae:.5f} RMSE={rmse:.5f} "
              "MSE/theory={mse_ratio:.4f} b_eff={effective_bits:.1f}".format(**row))


if __name__ == "__main__":
    main()
