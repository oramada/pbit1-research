from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from stochastic.hardware_equivalent import HardwareConfig, sample_devices, summarize_devices


def run(out_path: str = "results/hardware_equivalent.csv", seed: int = 2301,
        n_devices: int = 256, n_samples: int = 4096) -> list[dict[str, float | str]]:
    target = np.tile(np.linspace(0.02, 0.98, n_devices // 2), 2)
    base = HardwareConfig()
    rows = []
    # Isolate each nonideality before combining all sources of error.
    for voltage_bits in (4, 8, 12):
        for condition_index, (condition, changes) in enumerate((
            ("ideal", {"voltage_bits": 16, "calibration_slope_std": 0.0,
                       "calibration_offset_std": 0.0, "autocorrelation": 0.0,
                       "drift_std_volts_per_step": 0.0,
                       "readout_bit_flip_probability": 0.0}),
            ("voltage_only", {"voltage_bits": voltage_bits, "calibration_slope_std": 0.0,
                              "calibration_offset_std": 0.0, "autocorrelation": 0.0,
                              "drift_std_volts_per_step": 0.0,
                              "readout_bit_flip_probability": 0.0}),
            ("calibration_only", {"voltage_bits": 16, "autocorrelation": 0.0,
                                  "drift_std_volts_per_step": 0.0,
                                  "readout_bit_flip_probability": 0.0}),
            ("correlation_only", {"voltage_bits": 16, "calibration_slope_std": 0.0,
                                  "calibration_offset_std": 0.0,
                                  "drift_std_volts_per_step": 0.0,
                                  "readout_bit_flip_probability": 0.0}),
            ("drift_only", {"voltage_bits": 16, "calibration_slope_std": 0.0,
                            "calibration_offset_std": 0.0, "autocorrelation": 0.0,
                            "readout_bit_flip_probability": 0.0}),
            ("readout_only", {"voltage_bits": 16, "calibration_slope_std": 0.0,
                              "calibration_offset_std": 0.0, "autocorrelation": 0.0,
                              "drift_std_volts_per_step": 0.0}),
            ("combined", {"voltage_bits": voltage_bits}),
        )):
            config = replace(base, **changes)
            rng = np.random.default_rng(seed + voltage_bits * 10 + condition_index)
            observations = sample_devices(target, n_samples, config, rng)
            metrics = summarize_devices(target, observations)
            rows.append({"condition": condition, "voltage_bits": config.voltage_bits,
                         "n_devices": n_devices, "n_samples": n_samples,
                         "dac_bits_requested": voltage_bits,
                         "beta_per_volt": config.beta_per_volt,
                         "calibration_slope_std": config.calibration_slope_std,
                         "calibration_offset_std": config.calibration_offset_std,
                         "autocorrelation": config.autocorrelation,
                         "drift_std_volts_per_step": config.drift_std_volts_per_step,
                         "readout_bit_flip_probability": config.readout_bit_flip_probability,
                         **metrics})
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
        print(f"{row['condition']:16s} DAC={int(row['voltage_bits']):2d} bit "
              f"p-RMSE={row['probability_rmse']:.5f} "
              f"weight-RMSE={row['weight_rmse']:.5f}")
    Path("results/hardware_equivalent.md").write_text(
        "# Combined hardware-equivalent simulation\n\n"
        "`hardware_equivalent.csv` compares an ideal reference, voltage-only and "
        "single-nonideality ablations against a combined behavioral model with calibration slope/offset error, first-order temporal "
        "autocorrelation, voltage random-walk drift, and binary readout flips. The model "
        "uses 256 devices and 4096 samples each. The ideal reference uses a 16-bit DAC; "
        "voltage-only and combined rows use the DAC width named in each row. The model "
        "uses beta=4 V^-1, a +/-1.5 V control range, slope standard deviation 0.02, "
        "logit-offset standard deviation 0.02, lag-one rho=0.1, voltage drift standard "
        "deviation 1e-4 V/sample, and readout bit-flip probability 0.01. DAC widths are "
        "4, 8, and 12 bits. All numbers are "
        "illustrative and are not evidence about a physical device. Drift is modeled as "
        "a random walk without a restoring process, and probability changes under drift "
        "use a local-equilibrium Markov approximation.\n"
    )


if __name__ == "__main__":
    main()
