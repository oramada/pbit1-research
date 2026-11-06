from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stochastic.cross_correlated import (sample_cross_correlated,
                                         theoretical_dot_variance)


def run(out_path="results/cross_correlation_metrics.csv", seed=1414,
        n_trials=512):
    rng = np.random.default_rng(seed)
    dimensions = (16, 64, 256, 1024, 4096)
    samples_list = (1, 16, 64)
    common_modes = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75)
    rows = []
    # p=.5 and x_i=1/sqrt(d) isolate the effect of shared device noise.
    for dimension in dimensions:
        p = np.full(dimension, 0.5)
        x = np.full(dimension, 1.0 / np.sqrt(dimension))
        exact = 0.0
        for common_mode in common_modes:
            one_draw_variance = theoretical_dot_variance(p, x, common_mode)
            for n_samples in samples_list:
                estimates = np.empty(n_trials)
                chunk = max(1, min(64, n_trials, 4_000_000 // (dimension * n_samples)))
                for start in range(0, n_trials, chunk):
                    count = min(chunk, n_trials - start)
                    bits = sample_cross_correlated(
                        p, count * n_samples, common_mode, rng
                    ).reshape(count, n_samples, dimension)
                    spins = 2.0 * bits.astype(np.float64) - 1.0
                    estimates[start:start+count] = np.mean(spins @ x, axis=1)
                mse = float(np.mean((estimates - exact) ** 2))
                rows.append({
                    "dimension": dimension,
                    "common_mode_fraction": common_mode,
                    "samples": n_samples,
                    "empirical_mse": mse,
                    "theoretical_mse": one_draw_variance / n_samples,
                    "mse_ratio": mse / (one_draw_variance / n_samples)
                    if one_draw_variance else 1.0,
                    "rmse": float(np.sqrt(mse)),
                    "effective_variance_multiplier":
                        one_draw_variance / theoretical_dot_variance(p, x, 0.0),
                })
    # Validate marginals and pairwise correlation for homogeneous p=0.5 devices.
    calibration_rows = []
    p = np.full(8, 0.5)
    for common_mode in common_modes:
        bits = sample_cross_correlated(p, 100_000, common_mode, rng)
        corr = np.corrcoef(bits.astype(float), rowvar=False)
        off_diag = corr[np.triu_indices_from(corr, k=1)]
        calibration_rows.append({
            "common_mode_fraction": common_mode,
            "empirical_mean_probability": float(bits.mean()),
            "empirical_pair_correlation": float(off_diag.mean()),
            "pair_correlation_std": float(off_diag.std()),
        })
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calibration_path = path.with_name("cross_correlation_calibration.csv")
    with calibration_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)
    return rows, calibration_rows


if __name__ == "__main__":
    data, calibration = run()
    for row in calibration:
        print(row)
    print("selected dot-product cases:")
    for row in data:
        if row["dimension"] in (64, 1024, 4096) and row["samples"] == 16:
            print(row)
