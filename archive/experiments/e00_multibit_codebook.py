from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from weight_encodings.pbit_multibit import MultiPBit


def audit(bits: int, n_targets: int = 100_001) -> dict[str, float | int]:
    """Verify signed bipolar codebook coverage and quantify scalar quantization."""
    codec = MultiPBit(bits)
    targets = np.linspace(-1.0, 1.0, n_targets, dtype=np.float64)
    encoded = codec.encode(targets)
    decoded = codec.decode(codec.sample(encoded, n_samples=3,
                                        rng=np.random.default_rng(bits)))
    unique = np.unique(decoded)
    expected_n_levels = 1 << bits
    expected_levels = np.linspace(-1.0, 1.0, expected_n_levels)
    expected_steps = 2.0 / (expected_n_levels - 1)

    # Code-to-value mapping is independently reconstructed from the bipolar planes.
    spins, coeffs = encoded
    independent = np.sum(spins * coeffs, axis=-1)
    if not np.allclose(decoded, independent, atol=1e-14, rtol=0.0):
        raise AssertionError("decode disagrees with the bipolar positional expansion")
    if not np.allclose(unique, expected_levels, atol=1e-14, rtol=0.0):
        raise AssertionError("codebook is not the full uniformly spaced signed codebook")
    if not np.array_equal(spins[0], -np.ones(bits, dtype=np.int8)):
        raise AssertionError("-1 endpoint must encode as all-negative spins")
    if not np.array_equal(spins[-1], np.ones(bits, dtype=np.int8)):
        raise AssertionError("+1 endpoint must encode as all-positive spins")

    errors = decoded - targets
    # For uniform input on [-1,1], asymptotic MSE is step^2/12.
    return {
        "bits_per_weight": bits,
        "pbits_per_weight": bits,
        "codebook_levels": len(unique),
        "target_count": n_targets,
        "mean_bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "worst_abs_error": float(np.max(np.abs(errors))),
        "theoretical_half_step_bound": expected_steps / 2,
        "observed_step": float(np.min(np.diff(unique))),
        "expected_step": expected_steps,
        "endpoint_error": float(max(abs(errors[0]), abs(errors[-1]))),
    }


def main() -> None:
    rows = [audit(k) for k in (2, 4, 8)]
    path = Path("results/multibit_codebook.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("device count for N parameters: K=2 -> 2N, K=4 -> 4N, K=8 -> 8N")


if __name__ == "__main__":
    main()
