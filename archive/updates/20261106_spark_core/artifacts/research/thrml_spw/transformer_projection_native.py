"""Native sample/accumulate for a frozen pretrained attention projection.

This is one converted operator, not whole-model conversion. Output arrays are
later replayed through all downstream Transformer nonlinearities in Torch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .fashion_native import quantized
from .layer_budgets import write_csv
from .provenance import manifest
from .range_native import RangeNativeLinear


def main():
    source = Path("data/thrml_spw/transformer_projection/export.npz")
    data = np.load(source)
    prepared = json.loads(Path("results/thrml_spw/transformer_preparation/manifest.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == prepared["export_sha256"]
    out = Path("results/thrml_spw/transformer_projection_native")
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"model": prepared["config"]["model"], "model_revision": prepared["config"]["model_revision"],
                "target": prepared["config"]["target_module"], "converted_parameters": int(data["weight"].size),
                "converted_fraction": prepared["converted_fraction"],
                "mapping": "per-output lower+span*p, float32 metadata; one native logical node per selected matrix weight",
                "sample_budgets": [16, 64, 256], "sampling_seeds": list(range(5)), "windows_per_weight_bank_batch": 8,
                "key_domain": 930730, "input_export_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "arithmetic": "THRML Gibbs states and linear output accumulation before all downstream nonlinearities; JAX float32 external arithmetic; no weight reconstruction buffer",
                "scope": "Selective projection conversion only; embeddings, LayerNorm, other projections and output head deterministic. No physical performance or whole-model compression claim."}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    start = time.monotonic()
    layer = RangeNativeLinear(data["weight"], data["bias"])
    print(f"Native Transformer bank built: {layer.bank.logical_pbits} nodes, elapsed={time.monotonic()-start:.1f}s", flush=True)
    outputs, rows = {}, []
    q = layer.bank.realized_probabilities
    variance = layer.span**2*q*(1-q)
    for split_index, split in enumerate(("validation", "test")):
        inputs = data[f"{split}_inputs"]
        reference = data[f"{split}_reference_outputs"]
        outputs[split+"_FP32"] = reference
        outputs[split+"_zero_matrix"] = np.broadcast_to(data["bias"], reference.shape).copy()
        for bits in (4, 8):
            wq = quantized([data["weight"]], bits, "per_output")[0]
            outputs[f"{split}_affine{bits}"] = inputs@wq+data["bias"]
        expected_variance = float(np.mean(inputs.astype(np.float64)**2@variance))
        for t in protocol["sample_budgets"]:
            fn = layer.compiled(t)
            for seed in protocol["sampling_seeds"]:
                chunks = []
                for batch, first in enumerate(range(0, len(inputs), 8)):
                    key = jax.random.key(protocol["key_domain"])
                    for value in (split_index, seed, batch):
                        key = jax.random.fold_in(key, value)
                    x = inputs[first:first+8]
                    result = fn(key, jnp.asarray(x.reshape(-1, x.shape[-1])))
                    chunks.append(np.asarray(result).reshape(x.shape[0], x.shape[1], -1))
                y = np.concatenate(chunks)
                outputs[f"{split}_T{t}_seed{seed}"] = y
                rows.append({"split": split, "samples": t, "sampling_seed": seed,
                    "linear_mse": float(np.mean((y.astype(np.float64)-reference)**2)),
                    "predicted_sampling_mse": expected_variance/t,
                    "bank_batches": (len(inputs)+7)//8,
                    "free_node_updates": ((len(inputs)+7)//8)*t*len(layer.bank.free_indices),
                    "logical_weight_readouts": ((len(inputs)+7)//8)*t*layer.bank.logical_pbits})
            print(f"Native Transformer split={split} T={t} completed elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"linear_metrics.csv", rows)
    np.savez_compressed(out/"outputs.npz", **outputs)
    manifest(out/"manifest.json", protocol, {"controls": layer.controls(), "linear_metrics": rows,
             "elapsed_seconds": time.monotonic()-start,
             "language_quality_status": "Pending replay through unchanged downstream model; no language quality inferred from projection MSE alone."})


if __name__ == "__main__":
    main()
