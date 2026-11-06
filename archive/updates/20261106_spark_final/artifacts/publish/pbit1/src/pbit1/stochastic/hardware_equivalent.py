"""Composable behavioral model of imperfect programmable p-bits.

Every default here is illustrative and must be replaced by measured device data
before claiming hardware fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .voltage_quantization import probability_to_voltage, quantize_voltage, voltage_to_probability


@dataclass(frozen=True)
class HardwareConfig:
    beta_per_volt: float = 4.0
    voltage_min: float = -1.5
    voltage_max: float = 1.5
    voltage_bits: int = 8
    calibration_slope_std: float = 0.02
    calibration_offset_std: float = 0.02  # logit units
    autocorrelation: float = 0.1
    drift_std_volts_per_step: float = 1e-4
    readout_bit_flip_probability: float = 0.01

    def validate(self) -> None:
        if self.beta_per_volt <= 0:
            raise ValueError("beta_per_volt must be positive")
        if not 0 <= self.autocorrelation < 1:
            raise ValueError("autocorrelation must be in [0, 1)")
        if self.drift_std_volts_per_step < 0:
            raise ValueError("drift standard deviation cannot be negative")
        if not 0 <= self.readout_bit_flip_probability <= 0.5:
            raise ValueError("readout flip probability must be in [0, 0.5]")


def sample_devices(target_probabilities: np.ndarray, n_samples: int,
                   config: HardwareConfig, rng: np.random.Generator,
                   include_calibration: bool = True,
                   include_drift: bool = True,
                   include_readout_noise: bool = True) -> np.ndarray:
    """Draw measured binary time series for an array of programmed devices.

    Output shape is ``(n_samples, n_devices)``. For changing probabilities, the
    first-order Markov transition uses the current marginal p_t:
    P(z_t=1|z_{t-1}=0)=p_t(1-rho) and
    P(z_t=1|z_{t-1}=1)=p_t+rho(1-p_t).
    This preserves the requested stationary marginal when the bias is static and
    provides a defined local-equilibrium approximation under drift.
    """
    config.validate()
    p_req = np.asarray(target_probabilities, dtype=np.float64)
    if p_req.ndim != 1 or np.any((p_req < 0) | (p_req > 1)):
        raise ValueError("target_probabilities must be a vector in [0, 1]")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    voltage = probability_to_voltage(p_req, config.beta_per_volt)
    voltage = quantize_voltage(voltage, config.voltage_bits,
                               config.voltage_min, config.voltage_max)
    n_devices = len(p_req)
    if include_calibration:
        slope = rng.normal(1.0, config.calibration_slope_std, n_devices)
        offset = rng.normal(0.0, config.calibration_offset_std, n_devices)
    else:
        slope = np.ones(n_devices)
        offset = np.zeros(n_devices)

    drift = np.zeros(n_devices, dtype=np.float64)
    states = np.empty((n_samples, n_devices), dtype=np.uint8)
    for t in range(n_samples):
        if t and include_drift and config.drift_std_volts_per_step:
            drift += rng.normal(0.0, config.drift_std_volts_per_step, n_devices)
        p_t = voltage_to_probability(
            slope * config.beta_per_volt * (voltage + drift) + offset, 1.0)
        if t == 0:
            states[t] = rng.binomial(1, p_t)
        else:
            p_cond = np.where(states[t - 1] == 1,
                              p_t + config.autocorrelation * (1.0 - p_t),
                              p_t * (1.0 - config.autocorrelation))
            states[t] = rng.binomial(1, p_cond)
    if include_readout_noise and config.readout_bit_flip_probability:
        flips = rng.binomial(1, config.readout_bit_flip_probability, states.shape)
        states ^= flips.astype(np.uint8)
    return states


def summarize_devices(target_probabilities: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    """Summarize decoded probability and bipolar-weight error over devices."""
    target = np.asarray(target_probabilities, dtype=np.float64)
    measured = np.asarray(observations, dtype=np.float64).mean(axis=0)
    p_error = measured - target
    w_error = 2.0 * p_error
    return {
        "probability_bias": float(np.mean(p_error)),
        "probability_mae": float(np.mean(np.abs(p_error))),
        "probability_rmse": float(np.sqrt(np.mean(p_error ** 2))),
        "weight_rmse": float(np.sqrt(np.mean(w_error ** 2))),
        "weight_max_abs_error": float(np.max(np.abs(w_error))),
        "mean_observed_probability": float(np.mean(measured)),
    }

