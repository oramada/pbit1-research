"""Analytical conditional-variance/metadata forecasts, not sampled quality claims.

Use the current block's teacher inputs and one preserved native student trajectory
to quantify range localization. More groups reduce variance by moving information
into shared float metadata; that cost is counted explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    block = Path("results/thrml_spw/transformer_block0")
    out = Path("results/thrml_spw/block_range_analysis");out.mkdir(parents=True, exist_ok=True)
    weights = np.load(block/"weights.npz")
    rows, hashes = [], {}
    for stage in range(4):
        w = weights[f"weight_{stage}"].astype(np.float64)
        paths = {"teacher": block/"teacher"/f"validation_stage{stage}_inputs.npy",
                 "student_T64_seed0": block/"runs/validation_T64_seed0"/f"stage{stage}_inputs.npy"}
        for label, path in paths.items():
            x = np.load(path)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            mean_x2 = np.mean(x.astype(np.float64)**2, axis=(0, 1))
            for grouping in ("per_matrix", "per_output", "input_block16", "input_block32", "input_block64", "input_block128", "input_block256"):
                if grouping == "per_matrix":
                    lower, upper = w.min(), w.max()
                    variance = (upper-w)*(w-lower);groups = 1
                    free = int(np.sum((w > lower) & (w < upper)))
                elif grouping == "per_output":
                    lower, upper = w.min(0), w.max(0)
                    variance = (upper-w)*(w-lower);groups = w.shape[1]
                    free = int(np.sum((w > lower) & (w < upper)))
                else:
                    size = int(grouping.replace("input_block", ""))
                    if w.shape[0] % size:continue
                    shaped = w.reshape(-1, size, w.shape[1])
                    lower, upper = shaped.min(1, keepdims=True), shaped.max(1, keepdims=True)
                    variance = ((upper-shaped)*(shaped-lower)).reshape(w.shape)
                    groups = shaped.shape[0]*shaped.shape[2]
                    free = int(np.sum((shaped > lower) & (shaped < upper)))
                output_var = float(np.mean(mean_x2@variance))
                rows.append({"stage": stage, "inputs": label, "grouping": grouping,
                    "matrix_weights": w.size, "groups": groups, "metadata_bits_float32": 2*groups*32,
                    "metadata_bits_per_weight": 2*groups*32/w.size,
                    "ideal_free_nodes": free, "conditional_output_variance_single_draw": output_var,
                    "predicted_sampling_mse_at64": output_var/64,
                    "predicted_sampling_mse_at256": output_var/256})
    for row in rows:
        base = next(r for r in rows if r["stage"] == row["stage"] and r["inputs"] == row["inputs"] and r["grouping"] == "per_output")
        row["variance_fraction_of_per_output"] = row["conditional_output_variance_single_draw"]/base["conditional_output_variance_single_draw"]
    with (out/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result = {"scope": "Analytical ideal variance at fixed saved inputs; no new native trajectories or neural-quality conclusions. Smaller input groups add two float32 values per group. These forecasts do not replace the ongoing predeclared full-block gate.",
              "weights_sha256": hashlib.sha256((block/"weights.npz").read_bytes()).hexdigest(),
              "input_sha256": hashes, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": len(rows)}
    (out/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    for row in rows:
        if row["inputs"] == "teacher" and row["grouping"] in ("per_output", "input_block128"):
            print({k: row[k] for k in ("stage", "grouping", "variance_fraction_of_per_output", "metadata_bits_per_weight", "predicted_sampling_mse_at64")}, flush=True)


if __name__ == "__main__":main()
