"""Finite-bias positional-word controls targeting an adapted single-bit program.

Conversion only: this module neither samples THRML nor measures model quality.
The original lower/span metadata is unchanged. Each word bit requests field +/-4.
This calibrated internal format is distinct from the frozen original K8 study.
"""
import numpy as np


def expected_probability(codes):
    codes = np.asarray(codes)
    if codes.dtype != np.uint16 or codes.size == 0 or np.max(codes) > 4095:
        raise ValueError("Expected nonempty uint16 ACT12 field codes in [0,4095]")
    # Match programmed_worker.py: multiplication and addition in float32.
    fields = -4 + codes.astype(np.float32) * np.float32(8 / 4095)
    return 1 / (1 + np.exp(-2 * fields.astype(np.float64)))


def endpoint_probabilities():
    return float(1 / (1 + np.exp(8.0))), float(1 / (1 + np.exp(-8.0)))


def convert_codes(codes, bits):
    if bits not in (2, 4, 8):
        raise ValueError("Spatial word width must be 2, 4, or 8")
    target = expected_probability(codes)
    low, high = endpoint_probabilities()
    unit = np.clip((target - low) / (high - low), 0, 1)
    words = np.rint(unit * ((1 << bits) - 1)).astype(np.uint8)
    return words


def expected_word_probability(words, bits):
    if bits not in (2, 4, 8):
        raise ValueError("Spatial word width must be 2, 4, or 8")
    words = np.asarray(words)
    if words.dtype != np.uint8 or words.size == 0 or np.max(words) >= (1 << bits):
        raise ValueError("Expected nonempty uint8 words within declared width")
    low, high = endpoint_probabilities()
    return low + (high - low) * words.astype(np.float64) / ((1 << bits) - 1)
