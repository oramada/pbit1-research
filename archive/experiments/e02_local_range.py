from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from weight_encodings.single_pbit import SinglePBit


def grouping_ids(shape: tuple[int, ...], kind: str, block_size: int = 64) -> np.ndarray:
    flat = np.arange(np.prod(shape)).reshape(shape)
    if kind == "global":
        return np.zeros(shape, dtype=np.int64)
    if kind == "block":
        return flat // block_size
    if len(shape) == 2 and kind == "row":
        return np.broadcast_to(np.arange(shape[0])[:, None], shape)
    if len(shape) == 2 and kind == "column":
        return np.broadcast_to(np.arange(shape[1])[None, :], shape)
    raise ValueError(f"unsupported grouping {kind!r} for shape {shape}")


def main() -> None:
    rng = np.random.default_rng(12)
    weights = rng.normal(0, 0.08, size=(256, 256))
    rows = []
    groupings = [("global", grouping_ids(weights.shape, "global")),
                 ("output_channel", grouping_ids(weights.shape, "row")),
                 ("input_channel", grouping_ids(weights.shape, "column"))]
    groupings.extend((f"block{size}", grouping_ids(weights.shape, "block", size))
                     for size in (16, 32, 64, 128))
    for kind, ids in groupings:
        state = SinglePBit(eps=1e-6).encode(weights, ids)
        metadata_bytes = state.n_groups * 2 * 4  # center/radius stored as float32
        expected = (state.centers[state.group_ids] + state.radii[state.group_ids]
                    * (2 * state.probabilities - 1))
        rows.append({"grouping": kind, "range_alpha": 0.0, "groups": state.n_groups,
                     "metadata_bytes_float32": metadata_bytes,
                     "metadata_bits_per_weight": metadata_bytes * 8 / weights.size,
                     "reconstruction_rmse": float(np.sqrt(np.mean((expected - weights) ** 2))),
                     "residual_fraction": 0.0, "residual_storage_bytes_est": 0,
                     "mean_radius": float(np.mean(state.radii))})
        # Robust percentile clipping with sparse residual corrections.
        for alpha in (1e-4, 1e-3, 5e-3, 1e-2):
            clipped = weights.copy()
            for group in np.unique(ids):
                mask = ids == group
                low, high = np.quantile(weights[mask], [alpha, 1.0-alpha])
                clipped[mask] = np.clip(weights[mask], low, high)
            robust = SinglePBit(eps=1e-6).encode(clipped, ids)
            decoded = (robust.centers[robust.group_ids] + robust.radii[robust.group_ids]
                       * (2 * robust.probabilities - 1))
            residual_count = int(np.count_nonzero(weights != clipped))
            rows.append({"grouping": kind, "range_alpha": alpha,
                         "groups": robust.n_groups,
                         "metadata_bytes_float32": robust.n_groups * 8,
                         "metadata_bits_per_weight": robust.n_groups * 8 * 8 / weights.size,
                         "reconstruction_rmse": float(np.sqrt(np.mean((decoded - clipped) ** 2))),
                         "residual_fraction": residual_count / weights.size,
                         # Approximate each correction by a 32-bit index + float32 value.
                         "residual_storage_bytes_est": residual_count * 8,
                         "mean_radius": float(np.mean(robust.radii))})
    Path("results").mkdir(exist_ok=True)
    with open("results/local_range_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
