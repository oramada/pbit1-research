from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from weight_encodings.residual_pbit import ResidualPBit
from weight_encodings.single_pbit import SinglePBit


def empirical_curve(weights, encoded, kind, budgets, rng, trials=12):
    """Empirical mean squared error with a streamed Monte Carlo estimator."""
    max_t = max(budgets)
    totals = np.zeros((trials, *weights.shape), dtype=np.float32)
    result = {}
    for t in range(1, max_t + 1):
        b1 = rng.binomial(1, encoded.p_coarse, size=(trials, *weights.shape)).astype(np.float32)
        s1 = 2 * b1 - 1
        draw = encoded.center + encoded.radius * s1
        if kind == "residual":
            b2 = rng.binomial(1, encoded.p_residual, size=(trials, *weights.shape)).astype(np.float32)
            s2 = 2 * b2 - 1
            draw += encoded.radius * encoded.residual_scale * s2 * encoded.use_residual
        totals += draw
        if t in budgets:
            result[t] = float(np.mean((totals / t - weights[None, ...]) ** 2)) ** 0.5
    return result


def main():
    checkpoint = Path("data/checkpoints/mnist_seed0.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # The first large dense matrix is a reproducible representative layer.
    name = next(k for k, v in state.items() if k.endswith("weight") and v.ndim == 2)
    weights = state[name].numpy().astype(np.float64)
    low, high = weights.min(), weights.max()
    center, radius = (low + high) / 2, (high - low) / 2
    budgets = (1, 8, 64)
    rng = np.random.default_rng(20261026)
    output = []

    one = SinglePBit().encode(weights)
    # Benchmark one p-bit per scalar under the same tensor-local range.
    one_probs = one.probabilities
    target = weights
    one_rows = []
    for t in budgets:
        mse_theory = float(np.mean(4 * one.radii[one.group_ids]**2 *
                                   one_probs * (1-one_probs)) / t)
        one_rows.append({"encoding": "single_pbit", "samples": t,
                         "pbits_per_weight": 1.0, "expected_bias_rmse": float(np.sqrt(np.mean((
                             one.centers[one.group_ids] + one.radii[one.group_ids]*(2*one_probs-1)-weights)**2))),
                         "theoretical_estimator_rmse": np.sqrt(mse_theory),
                         "empirical_estimator_rmse": np.nan,
                         "quantization_rmse": 0.0})
    output.extend(one_rows)

    # Residual p-bits, at full occupancy and with two adaptive selection policies.
    adaptive_probe = ResidualPBit(residual_scale=0.25).encode(
        weights, np.ones(weights.shape, dtype=bool), center=center, radius=radius)
    base_var = 4 * radius**2 * one_probs * (1-one_probs)
    candidate_var = (4 * radius**2 * adaptive_probe.p_coarse * (1-adaptive_probe.p_coarse)
                     + 4 * (0.25*radius)**2 * adaptive_probe.p_residual *
                     (1-adaptive_probe.p_residual))
    variance_mask = candidate_var < base_var
    for scale in (0.5, 0.25, 0.125):
        residual = ResidualPBit(residual_scale=scale)
        policies = [("all", np.ones(weights.shape, dtype=bool))]
        if scale == 0.25:
            policies += [("variance_selected", variance_mask)]
            mask_rng = np.random.default_rng(20261027)
            policies += [("random_half", mask_rng.random(weights.shape) < 0.5)]
        for policy, mask in policies:
            encoded = residual.encode(weights, mask, center=center, radius=radius)
            var_coarse = 4 * radius**2 * encoded.p_coarse * (1-encoded.p_coarse)
            var_resid = (4 * (scale*radius)**2 * encoded.p_residual *
                         (1-encoded.p_residual) * mask)
            for t in budgets:
                expected = residual.expected(encoded)
                bias_mse = float(np.mean((expected - weights)**2))
                mse = bias_mse + float(np.mean(var_coarse + var_resid)) / t
                output.append({"encoding": f"residual_s{scale:g}_{policy}",
                               "samples": t,
                               "pbits_per_weight": residual.n_pbits(encoded) / weights.size,
                               "expected_bias_rmse": np.sqrt(bias_mse),
                               "theoretical_estimator_rmse": np.sqrt(mse),
                               "empirical_estimator_rmse": np.nan,
                               "quantization_rmse": np.sqrt(bias_mse)})

    # Monte Carlo verification for representative one-pbit and residual configurations.
    single_encoded = type("Encoded", (), {
        "p_coarse": one.probabilities,
        "p_residual": np.full(weights.shape, 0.5),
        "center": float(one.centers[0]),
        "radius": float(one.radii[0]),
        "residual_scale": 0.0,
        "use_residual": np.zeros(weights.shape, dtype=bool),
    })()
    single_emp = empirical_curve(weights, single_encoded, "single", budgets,
                                 np.random.default_rng(701))
    for row in output:
        if row["encoding"] == "single_pbit":
            row["empirical_estimator_rmse"] = single_emp[row["samples"]]
    residual_emp_state = ResidualPBit(0.25).encode(
        weights, variance_mask, center=center, radius=radius)
    residual_emp = empirical_curve(weights, residual_emp_state, "residual", budgets,
                                   np.random.default_rng(702))
    for row in output:
        if row["encoding"] == "residual_s0.25_variance_selected":
            row["empirical_estimator_rmse"] = residual_emp[row["samples"]]

    # K=4 symmetric quantization over the layer's local range.
    levels = 7
    scale4 = radius / levels
    quant = center + np.clip(np.rint((weights-center)/scale4), -levels, levels) * scale4
    output.append({"encoding": "K4_local_symmetric", "samples": 1,
                   "pbits_per_weight": 4.0, "expected_bias_rmse": float(np.sqrt(np.mean((quant-weights)**2))),
                   "theoretical_estimator_rmse": float(np.sqrt(np.mean((quant-weights)**2))),
                   "empirical_estimator_rmse": np.nan,
                   "quantization_rmse": float(np.sqrt(np.mean((quant-weights)**2)))})

    path = Path("results/residual")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"weight tensor={name}, shape={weights.shape}, range=({low:.5g},{high:.5g})")
    for row in output:
        print(row)


if __name__ == "__main__":
    main()
