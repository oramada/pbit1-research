from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stochastic.correlated import (calibrate_probability, effective_sample_size,
                                   sample_markov)


def main() -> None:
    rng = np.random.default_rng(31)
    p = np.linspace(0.05, 0.95, 1000)
    n_samples = 512
    rows = []
    for rho in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75):
        bits = sample_markov(np.broadcast_to(p, (128, 1000)), n_samples, rho, rng)
        means = bits.mean(axis=0)
        errors = means - p[None, :]
        rows.append({"rho": rho,
                     "theoretical_effective_samples": effective_sample_size(n_samples, rho),
                     "empirical_rmse": float(np.sqrt(np.mean(errors ** 2))),
                     "independent_rmse": float(np.sqrt(np.mean(p * (1-p) / n_samples)))})
    calibrated = calibrate_probability(p, sigma_a=0.05, sigma_b=0.05, rng=rng)
    calibration_rmse = float(np.sqrt(np.mean((calibrated - p) ** 2)))
    drift_sigma = 1e-3
    drift_limit = 0.05
    bias = np.zeros(1000)
    recalibrations = 0
    for _ in range(10000):
        bias += rng.normal(0.0, drift_sigma, size=bias.shape)
        crossed = np.abs(bias) >= drift_limit
        recalibrations += int(np.count_nonzero(crossed))
        bias[crossed] = 0.0
    Path("results").mkdir(exist_ok=True)
    with open("results/nonideality_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"one-device probability calibration RMSE (sigma_a=sigma_b=0.05): {calibration_rmse:.6f}")
    print(f"random-walk drift: sigma={drift_sigma}, threshold={drift_limit}, "
          f"recalibrations/device over 10000 steps={recalibrations / bias.size:.3f}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
