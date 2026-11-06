"""Exact field calibration for target marginals in coupled native THRML Ising banks.

For each tiny model, solve the fields so exact equilibrium marginals match the
requested per-spin probabilities. Then run the uncalibrated and calibrated fields
through the pinned THRML Gibbs simulator and report both marginal recovery and
remaining spatial/temporal correlation. This is a simulator experiment, not hardware
calibration or a claim that coupling has been removed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import root
from scipy.special import expit

from .correlations import (BETA, N_SAMPLES, N_WARMUP, SEEDS, SPACINGS,
                           TARGETS, enumerate_model, sample_native,
                           temporal_covariance, trajectory_stats)
from .provenance import manifest


OUT = Path("results/thrml_spw/coupling_calibration")
COUPLINGS = (0.35, 0.80)


def exact_moments(n: int, biases: np.ndarray, coupling: float):
    """Enumerate the exact Ising distribution and node-spin covariance."""
    edges = [(i, i + 1) for i in range(n - 1)]
    states = np.asarray(list(__import__("itertools").product([-1, 1], repeat=n)),
                        dtype=np.int8)
    energy = -BETA * (states @ biases)
    for i, j in edges:
        energy -= BETA * coupling * states[:, i] * states[:, j]
    mass = np.exp(-energy - np.max(-energy))
    pi = mass / mass.sum()
    mean_s = pi @ states
    cov_s = (states.T * pi) @ states - np.outer(mean_s, mean_s)
    return states, pi, mean_s, cov_s, edges


def moment_jacobian(cov_s: np.ndarray) -> np.ndarray:
    """Derivative d E[z_i]/d b_j = 2 beta Cov(z_i,z_j) = beta/2 Cov(s_i,s_j)."""
    return (BETA / 2.0) * cov_s


def calibrate_fields(n: int, target_p: np.ndarray, coupling: float):
    """Solve exact equilibrium mean constraints by Newton-style moment matching."""
    independent_fields = np.log(target_p) - np.log1p(-target_p)
    initial = independent_fields / (2.0 * BETA)

    def moments(fields):
        _, _, mean_s, cov_s, _ = exact_moments(n, fields, coupling)
        return (mean_s + 1.0) / 2.0, cov_s

    def residual(fields):
        marginal, _ = moments(fields)
        return marginal - target_p

    def jacobian(fields):
        _, cov_s = moments(fields)
        return moment_jacobian(cov_s)

    solution = root(residual, initial, jac=jacobian, method="hybr", options={"xtol": 1e-11})
    if not solution.success and np.max(np.abs(residual(solution.x))) > 1e-9:
        raise RuntimeError(f"field calibration failed: {solution.message}")
    fields = np.asarray(solution.x, dtype=np.float64)
    calibrated_marginals, cov_s = moments(fields)

    # Check the analytic covariance derivative against a centered finite difference.
    jac = moment_jacobian(cov_s)
    step = 1e-5
    finite_difference = np.column_stack([
        (moments(fields + np.eye(n)[j] * step)[0] -
         moments(fields - np.eye(n)[j] * step)[0]) / (2 * step)
        for j in range(n)
    ])
    jac_error = float(np.max(np.abs(jac - finite_difference)))
    return fields, calibrated_marginals, cov_s, int(solution.nfev), jac_error


def exact_transition_moments(n: int, fields: np.ndarray, coupling: float, spacing: int):
    """Reuse the exact transition builder for this set of Ising fields."""
    independent_prob = expit(2.0 * BETA * fields)
    states, _, pi, mean_s, cov_s, sweep, _, edges, _ = enumerate_model(
        n, independent_prob, coupling)
    transition = np.linalg.matrix_power(sweep, spacing)
    lag_cov = temporal_covariance(states, pi, mean_s, transition)
    return (states, pi, mean_s, cov_s, lag_cov, (mean_s + 1.0) / 2.0, edges)


def covariance_error_rows(empirical: np.ndarray, exact: np.ndarray):
    return json.dumps(empirical.tolist()), json.dumps(exact.tolist())


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    solution_rows, metric_rows = [], []
    raw = {}
    calibration_data = {}
    for n in (2, 3, 4):
        target_p = TARGETS[:n]
        for coupling in COUPLINGS:
            independent_fields = np.log(target_p) - np.log1p(-target_p)
            independent_fields /= 2.0 * BETA
            calibrated_fields, calibrated_p, calibrated_cov, nfev, jac_error = calibrate_fields(
                n, target_p, coupling)
            calibration_data[(n, coupling)] = (independent_fields, calibrated_fields)
            for policy, fields in (("independent_logit_fields", independent_fields),
                                   ("moment_matched_fields", calibrated_fields)):
                states, pi, mean_s, cov_s, _, exact_p, edges = exact_transition_moments(
                    n, fields, coupling, 1)
                solution_rows.append({
                    "n_spins": n, "coupling_J": coupling, "field_policy": policy,
                    "target_p": json.dumps(target_p.tolist()),
                    "fields_b": json.dumps(fields.tolist()),
                    "exact_marginal_p": json.dumps(exact_p.tolist()),
                    "marginal_error_vs_target": json.dumps((exact_p-target_p).tolist()),
                    "exact_spatial_cov_spin": json.dumps(cov_s.tolist()),
                    "exact_spatial_cov_bernoulli": json.dumps((cov_s/4).tolist()),
                    "root_nfev": nfev if policy == "moment_matched_fields" else 0,
                    "jacobian_max_finite_difference_error": jac_error if policy == "moment_matched_fields" else 0.0,
                })
                for spacing in SPACINGS:
                    states, pi, mean_s, cov_s, exact_lag_cov, exact_p, edges = exact_transition_moments(
                        n, fields, coupling, spacing)
                    traj = sample_native(n, fields, edges, coupling, spacing,
                                         SEEDS, N_SAMPLES, N_WARMUP)
                    raw[f"n{n}_J{coupling}_{policy}_d{spacing}"] = traj
                    mean_chain, empirical_spatial, empirical_lag, empirical_lag_corr, per_seed_means = trajectory_stats(traj)
                    seed_marginals = (per_seed_means + 1.0) / 2.0
                    for i in range(n):
                        seed_values = seed_marginals[:, i]
                        seed_sd = float(seed_values.std(ddof=1))
                        seed_mean = float(seed_values.mean())
                        lo = seed_mean - 1.96 * seed_sd / np.sqrt(len(SEEDS))
                        hi = seed_mean + 1.96 * seed_sd / np.sqrt(len(SEEDS))
                        metric_rows.append({
                            "n_spins": n, "coupling_J": coupling,
                            "field_policy": policy, "spacing_sweeps": spacing,
                            "node_i": i, "target_p": float(target_p[i]),
                            "exact_marginal_p": float(exact_p[i]),
                            "native_marginal_mean_over_seeds": seed_mean,
                            "native_marginal_sd_across_seeds": seed_sd,
                            "native_marginal_ci95_low": lo,
                            "native_marginal_ci95_high": hi,
                            "native_marginal_bias_vs_target": seed_mean - target_p[i],
                            "native_marginal_error_vs_exact": seed_mean - exact_p[i],
                            "per_seed_marginals": json.dumps(seed_values.tolist()),
                            "exact_spatial_cov_spin_row": json.dumps(cov_s[i].tolist()),
                            "native_spatial_cov_spin_row": json.dumps(empirical_spatial[i].tolist()),
                            "exact_lag1_cov_spin_row": json.dumps(exact_lag_cov[i].tolist()),
                            "native_lag1_cov_spin_row": json.dumps(empirical_lag[i].tolist()),
                            "exact_lag1_autocorrelation": float(exact_lag_cov[i,i] / max(cov_s[i,i], 1e-15)),
                            "native_lag1_autocorrelation": float(empirical_lag_corr[i]),
                            "pbits": n, "seeds": json.dumps(list(SEEDS)),
                            "n_samples_per_seed": N_SAMPLES, "warmup_sweeps": N_WARMUP,
                        })
            print(f"n={n} J={coupling:.2f} calibrated fields="
                  f"{np.round(calibrated_fields, 4).tolist()} max target err="
                  f"{np.max(np.abs(calibrated_p-target_p)):.2e} jac_fd_err={jac_error:.2e}",
                  flush=True)

    np.savez_compressed(OUT / "native_trajectories.npz", **raw)
    write_csv(OUT / "calibrated_fields_and_exact_moments.csv", solution_rows)
    write_csv(OUT / "native_seed_metrics.csv", metric_rows)

    max_solve_error = max(max(abs(v) for v in json.loads(r["marginal_error_vs_target"]))
                          for r in solution_rows if r["field_policy"] == "moment_matched_fields")
    max_native_cal_error = max(abs(r["native_marginal_error_vs_exact"])
                               for r in metric_rows if r["field_policy"] == "moment_matched_fields")
    calibration_report = [
        "# Coupled Ising field calibration",
        "",
        "Exact enumeration was used to calibrate the unary fields of 2-, 3-, and 4-spin",
        "nearest-neighbor Ising banks. Each target vector is drawn from the same interior",
        "probabilities as the uncorrelated p-bank. For nonzero coupling J, the original",
        "fields are b_i=logit(p_i)/(2 beta). Calibrated fields solve E[z_i]=p_i using",
        "the exact finite-state distribution and Jacobian dE[z_i]/db_j=2 beta Cov(z_i,z_j).",
        f"The largest exact marginal residual after root solving was {max_solve_error:.3e}.",
        "A centered finite-difference check of the analytic Jacobian is retained for every",
        "model in the exact-moments CSV.",
        "",
        f"For the pinned native THRML sampler, each field policy uses {len(SEEDS)} seeds,",
        f"{N_SAMPLES} observations per chain, {N_WARMUP} warmup sweeps, and recording spacings",
        f"{SPACINGS}. Native marginal uncertainty is summarized across independent seeds",
        "with mean +/- 1.96 SD/sqrt(5); per-seed marginal estimates are retained. Exact",
        "and empirical spatial covariance and lag-one temporal covariance/autocorrelation",
        "are reported side by side. The largest calibrated native marginal deviation from",
        f"its exact target was {max_native_cal_error:.4f}.",
        "",
        "Field calibration restores equilibrium node marginals in the exact model; it does",
        "not restore independence. The reported off-diagonal spatial covariance and",
        "temporal autocorrelation remain nonzero in coupled banks. Results are from the",
        "pinned THRML/JAX simulator only. They are not physical-device calibration,",
        "correlation, latency, energy, or performance claims.",
        "",
        "Files: `calibrated_fields_and_exact_moments.csv`, `native_seed_metrics.csv`,",
        "`native_trajectories.npz`, and `manifest.json`.",
    ]
    (OUT / "report.md").write_text("\n".join(calibration_report) + "\n")
    summary = {
        "models": len(solution_rows), "native_conditions": len(metric_rows),
        "seeds": list(SEEDS), "max_exact_calibration_residual": max_solve_error,
        "max_native_calibrated_marginal_error_vs_exact": max_native_cal_error,
        "elapsed_seconds": time.time()-start,
        "scope": "exact model and THRML simulator; no hardware calibration claim",
    }
    config = {"n_spins": [2,3,4], "target_probabilities": TARGETS.tolist(),
              "couplings": COUPLINGS, "spacings_full_sweeps": SPACINGS,
              "seeds": SEEDS, "n_samples": N_SAMPLES, "n_warmup": N_WARMUP,
              "beta": BETA, "field_solver": "scipy.optimize.root; exact enumeration; covariance Jacobian",
              "field_policies": ["independent_logit_fields", "moment_matched_fields"],
              "topology": "nearest-neighbor chain"}
    manifest(OUT / "manifest.json", config, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
