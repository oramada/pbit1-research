"""Matched stochastic-transition comparisons for temporal and spatial p-bit budgets.

Two different baselines are deliberately kept separate:
1. Positional K-bit bipolar quantization is a deterministic code after programming.
   It has K devices per weight and no inference-time stochastic transitions.
2. A K-device stochastic-replica estimator uses K independent p-bits at the same
   probability and averages their decoded values. This is a clean matched-transition
   comparison against one p-bit sampled K times, but it is not positional binary coding.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def scalar_resource_sweep(n_targets: int, n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    weights = np.linspace(-1.0, 1.0, n_targets)
    p = (weights + 1.0) / 2.0
    rows = []
    for k in (1, 2, 4):
        # K separate physical p-bits, each transitioned once per weight estimate.
        spatial = rng.binomial(1, p[None, :, None], size=(n_trials, n_targets, k))
        spatial_est = 2.0 * spatial.mean(axis=2) - 1.0
        # One physical p-bit transitioned K times. Transition count per scalar matches.
        temporal_count = rng.binomial(k, p[None, :], size=(n_trials, n_targets))
        temporal_est = 2.0 * temporal_count / k - 1.0
        exact_mse = float(np.mean(4.0 * p * (1.0 - p)) / k)
        for method, estimate, devices, transitions in (
            ("spatial_stochastic_replicas", spatial_est, k, k),
            ("single_temporal_pbit", temporal_est, 1, k),
        ):
            err = estimate - weights[None, :]
            rows.append({
                "comparison": "matched_transition_scalar",
                "method": method,
                "K_or_T": k,
                "devices_per_weight": devices,
                "transitions_per_weight_estimate": transitions,
                "mean_bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "theoretical_rmse": float(np.sqrt(exact_mse)),
                "empirical_mse_over_theory": float(np.mean(err * err) / exact_mse),
                "interpretation": "K independent identical-bias Bernoulli draws; not positional code",
            })
    return rows


def positional_quantization_sweep(n_targets: int):
    weights = np.linspace(-1.0, 1.0, n_targets)
    rows = []
    for k in (1, 2, 4):
        positive_levels = (1 << k) - 1
        q = np.rint(weights * positive_levels) / positive_levels
        err = q - weights
        rows.append({
            "comparison": "positional_deterministic_quantization",
            "method": "uniform_signed_K_bit_code",
            "K_or_T": k,
            "devices_per_weight": k,
            "transitions_per_weight_estimate": 0,
            "mean_bias": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "theoretical_rmse": float("nan"),
            "empirical_mse_over_theory": float("nan"),
            "interpretation": "deterministic positional code after programming; zero sampling transitions",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--out-dir", default="results/resource_match")
    args = parser.parse_args()
    rows = scalar_resource_sweep(args.targets, args.trials, args.seed)
    rows.extend(positional_quantization_sweep(args.targets))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matched_resource.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "README.md").open("w") as f:
        f.write("""# Matched stochastic-transition comparison\n\n""")
        f.write("The scalar stochastic comparison gives each method exactly K random transitions per weight estimate. Spatial replication uses K independent p-bits with the same bias and averages one observation from each. Temporal SPW uses one p-bit with K independent observations. These estimators have identical sampling distributions because both aggregate K independent Bernoulli observations. This confirms resource equivalence, not a performance advantage.\n\n")
        f.write("The positional K-bit bipolar expansion is reported separately as a deterministic uniform quantizer. It consumes K physical binary devices per parameter but has zero inference-time stochastic transitions after programming. It must not be described as a stochastic-replica baseline or as matched-transition evidence.\n\n")
        f.write(f"This run used {args.targets} uniformly spaced scalar weights and {args.trials} Monte Carlo repetitions per condition. Details are in `matched_resource.csv`.\n")
    for row in rows:
        if row["comparison"] == "matched_transition_scalar":
            print(f"{row['method']:30s} K={row['K_or_T']} transitions={row['transitions_per_weight_estimate']} "
                  f"RMSE={row['rmse']:.5f} theory={row['theoretical_rmse']:.5f} "
                  f"MSE/theory={row['empirical_mse_over_theory']:.4f}")
    print(f"saved {out_dir / 'matched_resource.csv'}")


if __name__ == "__main__":
    main()
