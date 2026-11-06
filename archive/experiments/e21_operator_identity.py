from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stochastic.correlated import sample_markov, effective_sample_size


def make_local_range_weights(rng: np.random.Generator, d_in: int, d_out: int):
    """Synthetic weights with per-output ranges, as in the SPW encoding."""
    w = rng.normal(0.0, 0.12, size=(d_in, d_out))
    # A weight matrix is (input, output), so each output column is its own range group.
    low = w.min(axis=0, keepdims=True)
    high = w.max(axis=0, keepdims=True)
    c = (low + high) / 2.0
    r = (high - low) / 2.0
    p = np.full_like(w, 0.5)
    np.divide(w - c + r, 2.0 * r, out=p, where=(r > 0))
    return w, c, r, p


def run(seed: int = 211, repetitions: int = 8,
        output: str = "results/operator_identity.csv") -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for d_in in (16, 64, 256):
        d_out = max(8, d_in // 2)
        w, c, r, p = make_local_range_weights(rng, d_in, d_out)
        for batch in (1, 8, 64):
            x = rng.normal(size=(batch, d_in)) / np.sqrt(d_in)
            exact = x @ w
            variance_one = np.sum((x[:, :, None] ** 2) * (r[None, :, :] ** 2)
                                  * (4.0 * p[None, :, :] * (1.0 - p[None, :, :])), axis=1)
            for rho in (0.0, 0.25, 0.75):
                for n_samples in (1, 4, 16, 64):
                    direct_sq_errors = []
                    reconstructed_sq_errors = []
                    equivalence_max = 0.0
                    equivalence_rmses = []
                    for _ in range(repetitions):
                        bits = sample_markov(p, n_samples, rho, rng)
                        spins = 2.0 * bits.astype(np.float64) - 1.0
                        sampled_weights = c[None, :, :] + r[None, :, :] * spins
                        # Path A: accumulate outputs from each full stochastic realization.
                        path_a = np.einsum("bi,tio->tbo", x, sampled_weights, optimize=True).mean(axis=0)
                        # Path B: reconstruct each matrix element from that same trajectory,
                        # then apply one deterministic matrix multiply.
                        reconstructed_w = sampled_weights.mean(axis=0)
                        path_b = x @ reconstructed_w
                        diff = path_a - path_b
                        equivalence_max = max(equivalence_max, float(np.max(np.abs(diff))))
                        equivalence_rmses.append(float(np.mean(diff * diff)))
                        direct_sq_errors.append(float(np.mean((path_a - exact) ** 2)))
                        reconstructed_sq_errors.append(float(np.mean((path_b - exact) ** 2)))
                    # Markov approximation supplied in the design; the output variance scales
                    # by T_eff^{-1}, while rho=0 reduces to the independent-sample result.
                    teff = effective_sample_size(n_samples, rho)
                    predicted_mse = float(np.mean(variance_one) / teff)
                    direct_mse = float(np.mean(direct_sq_errors))
                    rows.append({
                        "d_in": d_in, "d_out": d_out, "batch": batch,
                        "rho": rho, "samples": n_samples, "repetitions": repetitions,
                        "max_abs_direct_vs_reconstructed": equivalence_max,
                        "rms_direct_vs_reconstructed": float(np.sqrt(np.mean(equivalence_rmses))),
                        "direct_operator_mse": direct_mse,
                        "reconstructed_weight_mse": float(np.mean(reconstructed_sq_errors)),
                        "mse_ratio_operator_to_reconstruction": direct_mse / max(float(np.mean(reconstructed_sq_errors)), 1e-300),
                        "theory_mse_using_effective_samples": predicted_mse,
                        "empirical_to_theory_mse": direct_mse / max(predicted_mse, 1e-300),
                        "effective_samples": teff,
                    })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    results = run()
    max_diff = max(row["max_abs_direct_vs_reconstructed"] for row in results)
    max_mse_ratio_error = max(abs(row["mse_ratio_operator_to_reconstruction"] - 1.0)
                              for row in results)
    print(f"conditions={len(results)}; max direct/reconstruct discrepancy={max_diff:.3e}; "
          f"max relative MSE mismatch={max_mse_ratio_error:.3e}")
    print(f"saved results/operator_identity.csv")
