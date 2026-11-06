"""Idealized p-bit voltage control with finite DAC resolution.

This is a parameterized simulator, not a model fitted to a particular device.
"""

from __future__ import annotations

import numpy as np


def logit(probability: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def probability_to_voltage(probability: np.ndarray, beta: float,
                            eps: float = 1e-12) -> np.ndarray:
    """Invert p=sigmoid(beta*V) for beta > 0."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    return logit(probability, eps) / beta


def quantize_voltage(voltage: np.ndarray, n_bits: int, v_min: float,
                     v_max: float) -> np.ndarray:
    """Uniform endpoint-inclusive DAC quantization with clipping."""
    if n_bits < 1 or v_max <= v_min:
        raise ValueError("n_bits must be positive and v_max must exceed v_min")
    levels = (1 << n_bits) - 1
    v = np.clip(np.asarray(voltage, dtype=np.float64), v_min, v_max)
    return v_min + np.rint((v - v_min) * levels / (v_max - v_min)) * (v_max - v_min) / levels


def voltage_to_probability(voltage: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        raise ValueError("beta must be positive")
    x = np.clip(beta * np.asarray(voltage, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def quantized_probability(probability: np.ndarray, n_bits: int, beta: float,
                          v_min: float, v_max: float,
                          eps: float = 1e-12) -> np.ndarray:
    requested_v = probability_to_voltage(probability, beta, eps)
    return voltage_to_probability(quantize_voltage(requested_v, n_bits, v_min, v_max), beta)

