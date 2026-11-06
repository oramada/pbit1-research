"""Native THRML dot-product MSE under exact coupled spatial/temporal covariance.

For small calibrated Ising banks, this compares the empirical mean-square error of
sampled linear observables with the exact finite-state covariance calculation and a
naive independent-Bernoulli estimate. Samples all come from pinned THRML Gibbs chains.
This is a simulator study; no physical-device correlation or speed is inferred.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

from .correlations import (BETA, N_WARMUP, enumerate_model, sample_native)
from .coupling_calibration import calibrate_fields
from .provenance import manifest


OUT = Path("results/thrml_spw/correlated_dot")
SAMPLE_COUNTS = (1, 16, 64, 256)
MAX_SAMPLES = max(SAMPLE_COUNTS)
N_WARMUP = 512
N_SEEDS = 5
TRIALS_PER_SEED_GROUP = 200
N_TRIALS = N_SEEDS * TRIALS_PER_SEED_GROUP
COUPLINGS = (0.0, 0.35, 0.8)


def dot_vectors(n: int) -> dict[str, np.ndarray]:
    positive = np.ones(n, dtype=np.float64)
    alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    graded = alternating * np.arange(1, n + 1, dtype=np.float64)
    candidates = {"positive_uniform": positive,
                  "mixed_alternating": alternating,
                  "mixed_graded": graded}
    return {name: x / np.linalg.norm(x) for name, x in candidates.items()}


def exact_bernoulli_covariance_matrices(states: np.ndarray, pi: np.ndarray,
                                       transition: np.ndarray, max_lag: int):
    """Full Cov[z_i(0), z_j(lag)] matrices from exact P powers."""
    z = (states.astype(np.float64) + 1.0) / 2.0
    mean_z = pi @ z
    centered = z - mean_z
    covariances = [((centered.T * pi) @ centered)]
    q_power = np.eye(len(pi), dtype=np.float64)
    for _lag in range(1, max_lag):
        q_power = q_power @ transition
        covariances.append((centered.T * pi) @ (q_power @ centered))
    return np.asarray(covariances), mean_z


def exact_scalar_covariance(covariance_matrices: np.ndarray,
                            target_p: np.ndarray, x: np.ndarray, samples: int):
    """Exact Var(mean f) from full spatial/temporal covariance matrices."""
    covariances = np.einsum("i,kij,j->k", x, covariance_matrices[:samples], x)
    mu = float(np.dot(target_p, x))
    full_covariance_variance = (
        samples * covariances[0] +
        2.0 * sum((samples - lag) * covariances[lag]
                  for lag in range(1, samples))
    ) / (samples ** 2)
    naive_independent_variance = float(np.sum(x * x * target_p * (1.0 - target_p)) / samples)
    spatial_only_variance = float(x @ covariance_matrices[0] @ x)
    if not np.isclose(spatial_only_variance, covariances[0], atol=1e-12):
        raise AssertionError("observable variance disagrees with full spatial covariance")
    return mu, full_covariance_variance, naive_independent_variance, covariances, spatial_only_variance


def exact_warmup_tv(pi: np.ndarray, transition: np.ndarray, warmup: int) -> float:
    initial = np.zeros_like(pi)
    # enumerate_model uses lexicographic product([-1,+1]), so state 0 is all spins -1.
    initial[0] = 1.0
    at_warmup = initial @ np.linalg.matrix_power(transition, warmup)
    return float(0.5 * np.abs(at_warmup - pi).sum())


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows, warmup_rows = [], []
    raw = {}
    p_base = np.asarray([0.20, 0.35, 0.65, 0.80], dtype=np.float64)
    seeds = tuple(seed_group * 100_000 + trial
                  for seed_group in range(N_SEEDS)
                  for trial in range(TRIALS_PER_SEED_GROUP))

    for n in (2, 3, 4):
        target_p = p_base[:n]
        vectors = dot_vectors(n)
        for coupling in COUPLINGS:
            if coupling == 0:
                fields = np.log(target_p) - np.log1p(-target_p)
                fields /= 2.0 * BETA
            else:
                fields, exact_cal_p, _, _, _ = calibrate_fields(n, target_p, coupling)
                if not np.allclose(exact_cal_p, target_p, atol=1e-10):
                    raise AssertionError("moment-matched fields did not recover target p")
            independent_equivalent_p = expit(2.0 * BETA * fields)
            states, energies, pi, mean_s, cov_s, sweep, _, edges, _ = enumerate_model(
                n, independent_equivalent_p, coupling)
            exact_p = (mean_s + 1.0) / 2.0
            if coupling == 0 and not np.allclose(exact_p, target_p, atol=1e-10):
                raise AssertionError("zero-coupling p bank failed exact marginal check")
            tv = exact_warmup_tv(pi, sweep, N_WARMUP)
            covariance_matrices, _ = exact_bernoulli_covariance_matrices(
                states, pi, sweep, MAX_SAMPLES)
            raw[f"n{n}_J{coupling}_exact_cov_z_by_lag"] = covariance_matrices
            warmup_rows.append({
                "n_spins": n, "coupling_J": coupling,
                "calibrated_target_p": json.dumps(target_p.tolist()),
                "field_b": json.dumps(fields.tolist()),
                "warmup_sweeps": N_WARMUP, "exact_tv_from_all_minus_initial": tv,
            })
            # One maximum-length native sample run per model is prefix-shared by all T.
            trajectories = sample_native(n, fields, edges, coupling, 1,
                                         seeds, MAX_SAMPLES, N_WARMUP)
            raw[f"n{n}_J{coupling}_spins"] = trajectories
            # The exact blocked sweep transition includes the same even/odd block order
            # used by sample_native; no independent-spin surrogate is used.
            for vector_name, x in vectors.items():
                outputs = (trajectories.astype(np.float64) + 1.0) / 2.0
                outputs = np.einsum("ctn,n->ct", outputs, x)
                for samples in SAMPLE_COUNTS:
                    mu, exact_mse, naive_mse, covariances, spatial_var = exact_scalar_covariance(
                        covariance_matrices, exact_p, x, samples)
                    trial_means = outputs[:, :samples].mean(axis=1)
                    errors = trial_means - mu
                    squared_errors = errors ** 2
                    empirical_mse = float(squared_errors.mean())
                    mse_se = float(squared_errors.std(ddof=1) / np.sqrt(N_TRIALS))
                    rows.append({
                        "n_spins": n, "coupling_J": coupling,
                        "field_policy": "independent_logit" if coupling == 0 else "exact_marginal_calibration",
                        "input_vector": vector_name, "input_values": json.dumps(x.tolist()),
                        "samples_T": samples, "trials": N_TRIALS,
                        "seed_groups": N_SEEDS, "trials_per_seed_group": TRIALS_PER_SEED_GROUP,
                        "target_dot_expectation": mu,
                        "empirical_mean_of_trial_means": float(trial_means.mean()),
                        "empirical_bias": float(errors.mean()),
                        "empirical_mse": empirical_mse,
                        "empirical_mse_se": mse_se,
                        "empirical_mse_ci95_low": max(0.0, empirical_mse - 1.96 * mse_se),
                        "empirical_mse_ci95_high": empirical_mse + 1.96 * mse_se,
                        "exact_full_covariance_mse": exact_mse,
                        "naive_independent_variance": naive_mse,
                        "exact_to_naive_ratio": exact_mse / naive_mse if naive_mse else 0.0,
                        "empirical_to_exact_ratio": empirical_mse / exact_mse if exact_mse else 0.0,
                        "single_sample_spatial_variance": spatial_var,
                        "lag_covariances_lags_0_to_Tminus1": json.dumps(covariances.tolist()),
                        "warmup_sweeps": N_WARMUP,
                        "warmup_exact_tv": tv,
                    })
                    raw[f"n{n}_J{coupling}_{vector_name}_T{samples}_means"] = trial_means
            print(f"n={n} J={coupling:.2f} exact p={np.round(exact_p, 4).tolist()} "
                  f"warmup TV={tv:.2e}; native trajectories={trajectories.shape}", flush=True)

    np.savez_compressed(OUT / "raw_native_trajectories_and_dot_means.npz", **raw)
    write_csv(OUT / "dot_product_mse.csv", rows)
    write_csv(OUT / "warmup_stationarity.csv", warmup_rows)
    max_warmup_tv = max(row["exact_tv_from_all_minus_initial"] for row in warmup_rows)
    exact_naive_ratios = [row["exact_to_naive_ratio"] for row in rows]
    mean_abs_empirical_ratio_error = float(np.mean([
        abs(row["empirical_to_exact_ratio"] - 1.0) for row in rows
    ]))
    report = f"""# Native THRML correlated dot-product validation

## Design

This bounded validation uses exact-marginal-calibrated nearest-neighbor Ising banks
with n=2, 3, and 4 spins and J in {COUPLINGS}. For J=0, unary fields directly program
the target p bank; for J>0, fields come from exact finite-state moment matching so the
stationary marginals equal the same target probabilities. The native path is pinned
THRML Gibbs sampling with even/odd blocks, {N_WARMUP} warmup sweeps, and 1-sweep sample
spacing. It records T=1, 16, 64, and 256 prefixes from the same chain runs.

There are {N_TRIALS} independent native chains per n/J model, organized as
{N_SEEDS} groups of {TRIALS_PER_SEED_GROUP} seeds. Three unit-L2 dot-product inputs are
used: positive uniform, alternating mixed-sign, and graded mixed-sign. Raw binary
trajectories and all per-trial dot-product prefix means are stored in the NPZ archive.

## Exact reference

For a state observable f(s)=x^T z, z=(s+1)/2, exact stationary lag covariances are
computed from the finite-state blocked-sweep transition P. For each requested T, the
variance is [T C_0 + 2 sum for k=1..T-1 of (T-k) C_k] / T^2, where
C_k=x^T Cov(z(0),z(k)) x. Full n-by-n spatial and lagged cross-covariance matrices
for every lag through 255 are saved in the NPZ archive and contracted with each input
vector. The comparison baseline omits all off-diagonal and temporal terms and uses
sum_i x_i^2 p_i(1-p_i)/T. Exact MSE includes no model/weight bias; the
separately reported empirical bias and warmup bound expose finite-chain deviations.

The largest exact total-variation distance from the all-minus initial state after
{N_WARMUP} sweeps was {max_warmup_tv:.3e}; this provides a direct finite-state check
that the prescribed warmup is long enough for these models. Across all comparisons,
the mean absolute relative deviation of empirical MSE from the full-covariance exact
prediction was {mean_abs_empirical_ratio_error:.3f}. Per-condition MSE standard errors
and 95% normal intervals are included in `dot_product_mse.csv`. At J=0 the exact
full-covariance and naive independent estimates agree (ratio 1). Across coupled models
and input orientations, the exact/full-naive ratio spans
{min(exact_naive_ratios):.3f} to {max(exact_naive_ratios):.3f}; sign choices can cancel
or amplify covariance terms, so a single scalar penalty factor is not appropriate.

## Interpretation and scope

The exact-to-naive ratio quantifies how coupled spatial covariance and temporal Gibbs
correlation change the stochastic dot-product sampling cost after marginal calibration.
Positive and mixed-sign vectors can amplify or partially cancel the spatial covariance
terms. Calibrating marginals does not restore independent Bernoulli variance, and the
remaining temporal covariance can further change the sample-mean MSE. Numeric results
are reported per model/vector/budget in the CSV rather than collapsed into a hardware
claim.

All trajectories are from the pinned THRML/JAX simulator. Sweep counts and covariance
are properties of these algorithms and models; this is not a physical-chip correlation,
latency, energy, or speed measurement.
"""
    (OUT / "report.md").write_text(report)
    summary = {
        "models": len(warmup_rows), "dot_product_conditions": len(rows),
        "trials_per_model": N_TRIALS, "max_exact_warmup_tv": max_warmup_tv,
        "exact_to_naive_ratio_min": min(exact_naive_ratios),
        "exact_to_naive_ratio_max": max(exact_naive_ratios),
        "mean_abs_empirical_to_exact_mse_ratio_error": mean_abs_empirical_ratio_error,
        "elapsed_seconds": time.time() - start,
        "scope": "native THRML/JAX simulation only; no physical hardware claim",
    }
    config = {
        "n_spins": [2, 3, 4], "couplings": COUPLINGS,
        "target_p": p_base.tolist(), "field_policy": "exact moment matching for J>0",
        "sample_counts": SAMPLE_COUNTS, "warmup_sweeps": N_WARMUP,
        "spacing_sweeps": 1, "trials": N_TRIALS,
        "seed_groups": N_SEEDS, "trials_per_seed_group": TRIALS_PER_SEED_GROUP,
        "inputs": "unit-L2 positive uniform, alternating mixed-sign, graded mixed-sign",
        "exact_variance": "full finite-state spatial/temporal covariance from P powers",
        "baseline": "sum_i x_i^2 p_i(1-p_i)/T, independent samples",
    }
    manifest(OUT / "manifest.json", config, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
