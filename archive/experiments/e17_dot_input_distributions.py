"""Phase 6 input-distribution experiment with conditional variance checks.

For each dimension, draw a fixed target weight vector and a set of input vectors.
Sampling is generated coordinatewise as Binomial(T, p_i), equivalent to averaging
T independent Bernoulli p-bit states while reducing Monte Carlo work by a factor T.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DIMENSIONS = (16, 64, 256, 1024, 4096, 16384)
SAMPLE_COUNTS = (1, 4, 16, 64, 256)
N_TRIALS = 1000
SEED = 20261026


def load_mnist_activations(n_trials: int, dim: int, rng: np.random.Generator,
                           root: str) -> np.ndarray:
    """Tile true normalized MNIST pixel vectors to requested synthetic dot width.

    MNIST has 784 input activation coordinates. Tiling preserves actual observed
    pixel values while allowing the requested widths; this is explicitly a
    synthetic neural-like activation control, not a claim that these are hidden
    activations from a layer of width `dim`.
    """
    try:
        from torchvision.datasets import MNIST
        data = MNIST(root, train=True, download=False).data.numpy().astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"MNIST cache unavailable at {root}: {exc}") from exc
    # Rescale pixels to [0,1], then use vectors selected from distinct images.
    images = data.reshape(len(data), -1) / 255.0
    indices = rng.integers(0, len(images), size=n_trials)
    selected = images[indices]
    reps = (dim + selected.shape[1] - 1) // selected.shape[1]
    tiled = np.empty((n_trials, dim), dtype=np.float32)
    for row, shift in enumerate(rng.integers(0, selected.shape[1], size=n_trials)):
        rolled = np.roll(selected[row], int(shift))
        tiled[row] = np.tile(rolled, reps)[:dim]
    return tiled.astype(np.float64)


def make_inputs(kind: str, n_trials: int, dim: int, rng: np.random.Generator,
                mnist_root: str) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(size=(n_trials, dim))
    if kind == "unit_normalized":
        values = rng.normal(size=(n_trials, dim))
        return values / np.linalg.norm(values, axis=1, keepdims=True)
    if kind == "heavy_tailed_t3":
        # Student t(df=3) rescaled to unit variance, preserving heavy tails.
        return rng.standard_t(df=3, size=(n_trials, dim)) / np.sqrt(3.0)
    if kind == "mnist_pixel_activation_tiled":
        return load_mnist_activations(n_trials, dim, rng, mnist_root)
    raise ValueError(kind)


def run(n_trials: int = N_TRIALS, dimensions=DIMENSIONS,
        sample_counts=SAMPLE_COUNTS, seed: int = SEED,
        mnist_root: str = "data/torchvision",
        out_path: str = "results/dot_input_distributions.csv"):
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    input_kinds = ("gaussian", "unit_normalized", "heavy_tailed_t3",
                   "mnist_pixel_activation_tiled")
    for dim in dimensions:
        # Shared target weights permit paired input-distribution comparisons.
        weights = rng.uniform(-1.0, 1.0, size=dim)
        probabilities = (weights + 1.0) / 2.0
        coordinate_variance = 1.0 - weights**2
        for kind in input_kinds:
            x = make_inputs(kind, n_trials, dim, rng, mnist_root)
            exact = x @ weights
            x2_sum = np.sum(x * x, axis=1)
            for n_samples in sample_counts:
                errors = np.empty(n_trials, dtype=np.float64)
                # Bound random-array size to avoid large-memory spikes at d=16384.
                chunk_size = max(1, min(128, 2_000_000 // dim))
                for first in range(0, n_trials, chunk_size):
                    last = min(n_trials, first + chunk_size)
                    counts = rng.binomial(n_samples, probabilities,
                                          size=(last-first, dim))
                    sampled_w_mean = 2.0 * counts / n_samples - 1.0
                    sampled_y = np.sum(x[first:last] * sampled_w_mean, axis=1)
                    errors[first:last] = sampled_y - exact[first:last]
                empirical_mse = float(np.mean(errors**2))
                empirical_bias = float(np.mean(errors))
                theory_per_trial = np.sum(x * x * coordinate_variance[None, :], axis=1) / n_samples
                theory_mse = float(np.mean(theory_per_trial))
                rows.append({
                    "input_distribution": kind,
                    "dimension": dim,
                    "samples": n_samples,
                    "trials": n_trials,
                    "empirical_bias": empirical_bias,
                    "empirical_mae": float(np.mean(np.abs(errors))),
                    "empirical_rmse": float(np.sqrt(empirical_mse)),
                    "theoretical_rmse": float(np.sqrt(theory_mse)),
                    "empirical_mse": empirical_mse,
                    "theoretical_mse": theory_mse,
                    "mse_ratio": empirical_mse / theory_mse if theory_mse > 0 else (1.0 if empirical_mse == 0 else float("inf")),
                    "mean_input_l2": float(np.mean(np.sqrt(x2_sum))),
                    "std_exact_output": float(np.std(exact)),
                })
                print(f"{kind:30s} d={dim:5d} T={n_samples:3d} "
                      f"rmse={np.sqrt(empirical_mse):.5g} "
                      f"theory={np.sqrt(theory_mse):.5g} ratio={empirical_mse/theory_mse if theory_mse else float('nan'):.3f}",
                      flush=True)
            del x
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--mnist-root", default="data/torchvision")
    parser.add_argument("--out", default="results/dot_input_distributions.csv")
    args = parser.parse_args()
    run(n_trials=args.trials, mnist_root=args.mnist_root, out_path=args.out)


if __name__ == "__main__":
    main()
