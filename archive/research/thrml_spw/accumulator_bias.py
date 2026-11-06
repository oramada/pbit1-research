"""Negative control: rounding each sampled output can leave a bias floor."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .accumulator import FixedPointLinear
from .layer_budgets import write_csv
from .provenance import manifest


def main():
    out = Path("results/thrml_spw/accumulator_bias")
    out.mkdir(parents=True, exist_ok=True)
    weights = np.array([[.25], [.75]], dtype=np.float32)
    inputs = np.array([[.3, .3]], dtype=np.float32)
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    support = np.asarray(inputs@states.T).reshape(-1)
    target = float((inputs@weights).item())
    rows, raw = [], {"weights": weights, "inputs": inputs, "states": states}
    for frac in (0, 2, 4):
        layer = FixedPointLinear(weights, np.zeros(1, dtype=np.float32), 24, frac)
        p = layer.bank.realized_probabilities.reshape(-1).astype(np.float64)
        mass = np.prod(np.where(states == 1, p, 1-p), axis=1)
        rounded = np.rint(support*(1 << frac))/(1 << frac)
        mean = float(mass@rounded)
        variance = float(mass@(rounded-mean)**2)
        for t in (16, 256, 4096):
            keys = jax.random.split(jax.random.key(793000+frac), 1000)
            fn = layer.compiled(t)
            outputs, counts = jax.jit(jax.vmap(lambda k: fn(k, jnp.asarray(inputs))))(keys)
            values = np.asarray(outputs).reshape(-1)
            assert not np.any(np.asarray(counts))
            error_squared = (values.astype(np.float64)-target)**2
            theory = (mean-target)**2+variance/t
            se = float(error_squared.std(ddof=1)/np.sqrt(len(values)))
            discrepancy = abs(float(error_squared.mean())-theory)/se
            assert discrepancy < 6
            rows.append({"fractional_bits": frac, "samples": t, "trials": len(values),
                         "deterministic_target": target, "exact_rounded_expectation": mean,
                         "asymptotic_bias": mean-target, "empirical_mean": float(values.mean()),
                         "empirical_mse": float(error_squared.mean()), "exact_mse": theory,
                         "empirical_mse_se": se, "mse_standard_error_units": discrepancy})
            raw[f"frac{frac}_T{t}"] = values
        print(f"fractional_bits={frac} target={target:.6f} rounded_expectation={mean:.6f} bias={mean-target:.6f}", flush=True)
    write_csv(out/"metrics.csv", rows)
    np.savez_compressed(out/"sample_means.npz", **raw)
    result = {"rows": rows,
              "interpretation": "Rounding increments before averaging changes their expectation. With no clipping the absolute mean shift is at most half a lattice unit, but does not generally vanish as T increases. This applies to modeled arithmetic, not to p-bit representation correctness."}
    (out/"REPORT.md").write_text("# Increment-rounding negative control\n\n"
        "Two independent native p-bits encode probabilities 0.25 and 0.75. "
        "The input [0.3,0.3] has deterministic dot product 0.3. Rounding each sampled "
        "dot product to integer increments yields exact limiting expectation 0.1875, "
        "not 0.3. Four fractional bits instead yield expectation 0.3125.\n\n"
        "Each of nine format/budget conditions used 1,000 actual THRML chains. "
        "Empirical MSE agrees within six estimated standard errors with "
        "bias squared plus rounded-increment variance divided by T. No accumulator "
        "saturation occurred. More temporal samples cannot remove this rounding "
        "bias. This negative control limits interpretation of the neural accumulator "
        "results; it does not invalidate probability encoding. See metrics.csv and "
        "raw sample means.\n")
    manifest(out/"manifest.json", {"fractional_bits": [0, 2, 4], "samples": [16, 256, 4096],
             "trials": 1000, "bits": 24, "weights": weights.tolist(), "inputs": inputs.tolist()}, result)


if __name__ == "__main__":
    main()
