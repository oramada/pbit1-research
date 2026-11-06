from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path("data/checkpoints/mnist_seed0.pt")
OUTPUT = Path("results/range_granularity/mnist_seed0.csv")


def get_weights():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    return [(name, value.numpy().astype(np.float64))
            for name, value in state.items() if name.endswith(".weight")]


def group_ids_by_strategy(tensors, strategy: str, block_size: int = 64) -> list[np.ndarray]:
    """Return contiguous group identifiers, globally unique across tensors."""
    ids_by_tensor = []
    next_group = 0
    total_offset = 0
    for _name, w in tensors:
        if strategy == "global":
            local = np.zeros(w.shape, dtype=np.int64)
        elif strategy == "layer":
            local = np.zeros(w.shape, dtype=np.int64)
        elif strategy == "output_channel":
            shape = (w.shape[0],) + (1,) * (w.ndim - 1)
            local = np.broadcast_to(np.arange(w.shape[0]).reshape(shape), w.shape)
        elif strategy == "input_channel":
            if w.ndim < 2:
                local = np.zeros(w.shape, dtype=np.int64)
            else:
                shape = [1] * w.ndim
                shape[1] = w.shape[1]
                local = np.broadcast_to(np.arange(w.shape[1]).reshape(shape), w.shape)
        elif strategy == "block":
            local = np.arange(w.size, dtype=np.int64).reshape(w.shape) // block_size
        else:
            raise ValueError(strategy)

        if strategy == "global":
            # Every model parameter shares one center and one radius.
            ids_by_tensor.append(np.zeros(w.shape, dtype=np.int64))
            total_offset += w.size
            continue

        local = np.asarray(local, dtype=np.int64)
        local_unique, remapped = np.unique(local, return_inverse=True)
        remapped = remapped.reshape(w.shape) + next_group
        ids_by_tensor.append(remapped)
        next_group += len(local_unique)
        total_offset += w.size
    return ids_by_tensor


def analyze_strategy(tensors, strategy: str, block_size: int = 64) -> dict:
    gids_by_tensor = group_ids_by_strategy(tensors, strategy, block_size)
    all_w = np.concatenate([w.ravel() for _, w in tensors])
    all_ids = np.concatenate([g.ravel() for g in gids_by_tensor])
    n_groups = int(all_ids.max()) + 1 if all_ids.size else 0
    centers = np.zeros(n_groups, dtype=np.float64)
    radii = np.zeros(n_groups, dtype=np.float64)
    ps = np.zeros_like(all_w)
    for g in range(n_groups):
        mask = all_ids == g
        lo, hi = all_w[mask].min(), all_w[mask].max()
        centers[g] = (lo + hi) / 2
        radii[g] = (hi - lo) / 2
        if radii[g] > 0:
            ps[mask] = 0.5 * ((all_w[mask] - centers[g]) / radii[g] + 1.0)
        else:
            ps[mask] = 0.5
    p_safe = np.clip(ps, 1e-6, 1 - 1e-6)
    expected = centers[all_ids] + radii[all_ids] * (2 * p_safe - 1)
    single_var = 4 * radii[all_ids] ** 2 * p_safe * (1 - p_safe)
    total_weights = all_w.size
    metadata_bytes = 8 * n_groups  # float32 center and radius per group
    result = {
        "strategy": strategy if strategy != "block" else f"block{block_size}",
        "groups": n_groups,
        "weights": total_weights,
        "metadata_bytes_float32": metadata_bytes,
        "metadata_bits_per_weight": 8 * metadata_bytes / total_weights,
        "mean_radius": float(np.mean(radii)),
        "median_radius": float(np.median(radii)),
        "max_radius": float(np.max(radii)),
        "mean_abs_reconstruction_bias": float(np.mean(np.abs(expected - all_w))),
        "fraction_clipped_probabilities": float(np.mean((ps <= 1e-6) | (ps >= 1 - 1e-6))),
    }
    for T in (1, 16, 64, 256, 1024):
        var = single_var / T
        result[f"rmse_T{T}"] = float(np.sqrt(np.mean(var + (expected - all_w) ** 2)))
        result[f"max_std_T{T}"] = float(np.sqrt(np.max(var)))
    return result


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing trained MNIST checkpoint: {CHECKPOINT}")
    tensors = get_weights()
    strategies = [("global", 0), ("layer", 0), ("output_channel", 0),
                  ("input_channel", 0)]
    strategies += [("block", n) for n in (16, 32, 64, 128)]
    rows = [analyze_strategy(tensors, strategy, block)
            for strategy, block in strategies]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
