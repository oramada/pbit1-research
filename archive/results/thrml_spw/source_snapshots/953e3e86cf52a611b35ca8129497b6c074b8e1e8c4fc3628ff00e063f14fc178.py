"""Confirm independently validation-chosen controls, sums, and layer budgets together.

This factorial confirmation retains all outcomes; it does not tune configurations
on test results. The official THRML Gibbs sampler drives every stochastic branch.
"""
from __future__ import annotations

import hashlib
import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .accumulator import FixedPointLinear
from .control_bank import ControlledProbabilityBank
from .layer_budgets import metrics, write_csv
from .positive_inference import NativeLinear
from .provenance import manifest


def run(layers, inputs, model, seed, domain, budgets, fixed):
    functions = [layer.compiled(t) for layer, t in zip(layers, budgets)]
    outputs, counts = [], np.zeros((2, 2), dtype=np.int64)
    for batch, first in enumerate(range(0, len(inputs), 32)):
        key = jax.random.key(domain)
        for value in (model, seed, batch):
            key = jax.random.fold_in(key, value)
        value = jnp.asarray(inputs[first:first+32])
        for i, (fn, k) in enumerate(zip(functions, jax.random.split(key))):
            if fixed:
                value, counters = fn(k, value)
                counts[i] += np.asarray(counters, dtype=np.int64)
            else:
                value = fn(k, value)
            if i == 0:
                value = jax.nn.relu(value)
        outputs.append(np.asarray(value))
    return np.concatenate(outputs), counts


def main():
    root = Path("results/thrml_spw")
    paths = {"allocation": root/"layer_budgets/selection.json",
             "accumulator": root/"accumulator_precision/selection.json",
             "control": root/"control_neural/selection.json"}
    selections = {name: json.loads(path.read_text())["selected"] for name, path in paths.items()}
    t0, t1 = selections["allocation"]["T0"], selections["allocation"]["T1"]
    bits, frac = selections["accumulator"]["bits"], selections["accumulator"]["fractional_bits"]
    if bits == 32:
        raise RuntimeError("No fixed-point format passed validation; this planned factorial requires a selected fixed-point format")
    field_bits, bound = selections["control"]["field_bits"], selections["control"]["field_bound"]
    domain = json.loads((root/"layer_budgets/protocol.json").read_text())["test_key_domain"]
    out = root/"joint_efficiency"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"selections": selections, "selection_uses_test_labels": False,
                "selection_sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()},
                "trained_model_seeds": list(range(5)), "sampling_seeds_per_model": list(range(5)),
                "test_examples": 10000, "test_domain": domain, "batch_size": 32,
                "comparisons": "2x2 factorial control/accumulator precision at selected layer budgets, plus original uniform256 reference; all outcomes retained without further tuning",
                "baseline_reuse": "Unquantized branches reuse exact parent logits with identical weights, keys, examples and schedules",
                "scope": "CPU/JAX observers and dimensionless field quantization; storage counts hypothetical, no measured chip energy/latency or Python memory compression"}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    data = np.load("data/thrml_spw/positive_mnist/evaluation_data.npz")
    parent = np.load(root/"layer_budgets/test_logits.npz")
    inputs, labels = data["test_inputs"], data["test_labels"]
    rows, raw, controls, start = [], {"labels": labels}, {}, time.monotonic()
    conditions = [("field_float32_sum_float32", False, False),
                  ("field_quantized_sum_float32", True, False),
                  ("field_float32_sum_fixed", False, True),
                  ("field_quantized_sum_fixed", True, True)]
    for model in range(5):
        z = np.load(f"data/thrml_spw/positive_mnist/seed{model}.npz")
        w, b = [z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)]
        for name, quantized, fixed in conditions:
            layers = []
            for wi, bi in zip(w, b):
                layer = FixedPointLinear(wi, bi, bits, frac) if fixed else NativeLinear(wi, bi)
                if quantized:
                    layer.bank = ControlledProbabilityBank(wi, field_bits, bound)
                layers.append(layer)
            controls[f"model{model}_{name}"] = [layer.bank.controls() for layer in layers]
            for seed in range(5):
                if not quantized and not fixed:
                    logits = parent[f"model{model}_seed{seed}_T{t0}_{t1}"]
                    counts = np.zeros((2, 2), dtype=np.int64)
                else:
                    logits, counts = run(layers, inputs, model, seed, domain, (t0, t1), fixed)
                uniform = parent[f"model{model}_seed{seed}_T256_256"]
                score, reference = metrics(logits, labels), metrics(uniform, labels)
                raw[f"model{model}_seed{seed}_{name}"] = logits
                free = [len(layer.bank.free_indices) for layer in layers]
                rows.append({"model_seed": model, "sampling_seed": seed, "configuration": name,
                    "T0": t0, "T1": t1, **score,
                    "accuracy_delta_vs_uniform256": score["accuracy"]-reference["accuracy"],
                    "logical_reads_per_isolated_example": 2048*t0+320*t1,
                    "free_updates_per_isolated_example": free[0]*t0+free[1]*t1,
                    "dependent_layer_sweeps": t0+t1,
                    "increment_clips": int(counts[:, 0].sum()), "sum_clips": int(counts[:, 1].sum())})
            print(f"joint confirmation model={model} condition={name} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", rows)
    np.savez_compressed(out/"test_logits.npz", **raw)
    summary = []
    for name, _, _ in conditions:
        a, d = [], []
        group = [r for r in rows if r["configuration"] == name]
        for model in range(5):
            per_model = [r for r in group if r["model_seed"] == model]
            a.append(np.mean([r["accuracy"] for r in per_model]))
            d.append(np.mean([r["accuracy_delta_vs_uniform256"] for r in per_model]))
        summary.append({"configuration": name, "accuracy_mean": float(np.mean(a)),
                        "accuracy_ci95_margin": float(1.96*np.std(a, ddof=1)/np.sqrt(5)),
                        "paired_accuracy_delta_vs_uniform256": float(np.mean(d)),
                        "paired_delta_ci95_margin": float(1.96*np.std(d, ddof=1)/np.sqrt(5)),
                        "total_increment_clips": sum(r["increment_clips"] for r in group),
                        "total_sum_clips": sum(r["sum_clips"] for r in group)})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", protocol, {"summary": summary, "controls": controls,
                                           "elapsed_seconds": time.monotonic()-start})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
