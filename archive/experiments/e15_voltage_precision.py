from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stochastic.voltage_quantization import quantized_probability


def run(out_path: str = "results/voltage_precision.csv", seed: int = 1501,
        beta: float = 4.0, v_min: float = -1.5, v_max: float = 1.5,
        n_probability_points: int = 10001, n_trials: int = 2000) -> list[dict[str, float]]:
    """Sweep DAC precision for a logistic p-bit transfer curve.

    beta=4 V^-1 and +/-1.5 V are illustrative parameters only. The first two
    metrics characterize deterministic control error; the Monte Carlo metrics
    additionally include Bernoulli sampling with the specified finite sample count.
    """
    rng = np.random.default_rng(seed)
    requested = np.linspace(0.001, 0.999, n_probability_points)
    rows = []
    for bits in (4, 6, 8, 10, 12):
        realized = quantized_probability(requested, bits, beta, v_min, v_max)
        calibration_error = realized - requested
        # Map a bipolar [-1,+1] scalar through p=(w+1)/2.
        weight_error = 2.0 * calibration_error
        rows.append({
            "voltage_bits": bits,
            "beta_per_volt": beta,
            "voltage_min": v_min,
            "voltage_max": v_max,
            "probability_rmse": float(np.sqrt(np.mean(calibration_error ** 2))),
            "probability_mae": float(np.mean(np.abs(calibration_error))),
            "probability_max_abs_error": float(np.max(np.abs(calibration_error))),
            "weight_rmse_from_control_only": float(np.sqrt(np.mean(weight_error ** 2))),
            "fraction_voltage_saturated": float(np.mean(
                (requested < 1.0 / (1.0 + np.exp(beta * v_max))) |
                (requested > 1.0 / (1.0 + np.exp(-beta * v_max))))),
            "sample_count": 256,
            "monte_carlo_weight_rmse": float("nan"),
            "total_weight_rmse": float("nan"),
        })
        # Stratified sweep of target probabilities, then finite Bernoulli estimation.
        p_realized = quantized_probability(requested, bits, beta, v_min, v_max)
        trial_indices = rng.integers(0, n_probability_points, size=n_trials)
        p_target = requested[trial_indices]
        p_device = p_realized[trial_indices]
        samples = rng.binomial(256, p_device)
        w_est = 2.0 * samples / 256.0 - 1.0
        w_target = 2.0 * p_target - 1.0
        rows[-1]["monte_carlo_weight_rmse"] = float(np.sqrt(np.mean((w_est - (2*p_device-1)) ** 2)))
        rows[-1]["total_weight_rmse"] = float(np.sqrt(np.mean((w_est - w_target) ** 2)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    rows = run()
    for row in rows:
        print(f"{int(row['voltage_bits']):2d} bits: p-RMSE={row['probability_rmse']:.6g}, "
              f"control-only w-RMSE={row['weight_rmse_from_control_only']:.6g}, "
              f"total 256-sample w-RMSE={row['total_weight_rmse']:.6g}")
    Path("results/voltage_precision.md").write_text(
        "# Voltage precision sweep\n\n"
        "The results in `voltage_precision.csv` use a logistic response with "
        "illustrative beta=4 V^-1, a uniform DAC over [-1.5, 1.5] V, and 256 "
        "Bernoulli observations. These are simulator assumptions, not measured device "
        "parameters. Control quantization error is separated from finite-sample error.\n"
    )


if __name__ == "__main__":
    main()
