"""Validation-only choice of fixed-point accumulator precision, native sampling."""
from __future__ import annotations

import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .accumulator import FixedPointLinear
from .layer_budgets import metrics, write_csv
from .positive_inference import NativeLinear
from .provenance import manifest


def run(layers, inputs, model, seed, domain, fixed, t=256):
    outputs, counts = [], np.zeros((2, 2), dtype=np.int64)
    compiled = [layer.compiled(t) for layer in layers]
    for batch, first in enumerate(range(0, len(inputs), 32)):
        key = jax.random.key(domain)
        for value in (model, seed, batch):
            key = jax.random.fold_in(key, value)
        keys = jax.random.split(key)
        value = jnp.asarray(inputs[first:first+32])
        for i, (fn, key) in enumerate(zip(compiled, keys)):
            if fixed:
                value, counters = fn(key, value)
                counts[i] += np.asarray(counters, dtype=np.int64)
            else:
                value = fn(key, value)
            if i == 0:
                value = jax.nn.relu(value)
        outputs.append(np.asarray(value))
    return np.concatenate(outputs), counts


def main():
    out = Path("results/thrml_spw/accumulator_precision")
    out.mkdir(parents=True, exist_ok=True)
    data = np.load("data/thrml_spw/positive_mnist/evaluation_data.npz")
    candidates = [(16, 0), (16, 2), (16, 4), (16, 8), (24, 4), (24, 8), (24, 12)]
    configurations = {"float32": None, **{f"int{b}_frac{f}": (b, f) for b, f in candidates}}
    protocol = {
        "samples_per_layer": 256, "candidates": candidates,
        "trained_model_seeds": list(range(5)), "validation_sampling_seeds": [0],
        "test_sampling_seeds": list(range(5)), "batch_size": 32,
        "validation_domain": 620730, "test_domain": 640730,
        "selection_rule": "For each total width choose fractional bits with lowest mean validation CE, then select smallest width with mean validation accuracy drop <=0.002 and CE increase <=0.01 relative to float32. If none passes retain float32.",
        "stress_control": "int16_frac8 is tested regardless of selection to expose saturation; no test-based choice",
        "rounding": "Nearest-even of float32 dot-product increments to a signed fixed-point lattice; increment and integer running sum saturated separately. Divide by T and lattice scale, then add float32 neural bias.",
        "scope": "Only the output accumulator is quantized. Controls, dot products, final division, biases and nonlinearities use float32. Integer containers use int32 in software; B-bit storage is hypothetical.",
        "selection_uses_test_labels": False,
        "interval": "Normal-approximation 1.96*SD/sqrt(5) across five trained-model means after averaging five sampling seeds; paired model differences for comparisons",
    }
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    start, models, rows = time.monotonic(), {}, []
    val_raw = {"labels": data["validation_labels"]}
    for model in range(5):
        z = np.load(f"data/thrml_spw/positive_mnist/seed{model}.npz")
        w, b = [z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)]
        models[model] = (w, b)
        for name, config in configurations.items():
            layers = [FixedPointLinear(wi, bi, *config) if config else NativeLinear(wi, bi) for wi, bi in zip(w, b)]
            logits, counts = run(layers, data["validation_inputs"], model, 0, protocol["validation_domain"], config is not None)
            val_raw[f"model{model}_{name}"] = logits
            rows.append({"model_seed": model, "configuration": name, **metrics(logits, data["validation_labels"]),
                         "layer0_increment_clips": int(counts[0, 0]), "layer0_sum_clips": int(counts[0, 1]),
                         "layer1_increment_clips": int(counts[1, 0]), "layer1_sum_clips": int(counts[1, 1])})
        print(f"accumulator validation model={model} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"validation_metrics.csv", rows)
    np.savez_compressed(out/"validation_logits.npz", **val_raw)
    profiles = {name: {"configuration": name, "bits": config[0] if config else 32,
                    "fractional_bits": config[1] if config else None,
                    "accuracy": float(np.mean([r["accuracy"] for r in rows if r["configuration"] == name])),
                    "cross_entropy": float(np.mean([r["cross_entropy"] for r in rows if r["configuration"] == name]))}
                for name, config in configurations.items()}
    base = profiles["float32"]
    choices = []
    for bits in (16, 24):
        best = min((r for r in profiles.values() if r["bits"] == bits), key=lambda r: (r["cross_entropy"], r["fractional_bits"]))
        choices.append(best)
    passing = [r for r in choices if base["accuracy"]-r["accuracy"] <= .002 and r["cross_entropy"]-base["cross_entropy"] <= .01]
    selected = min(passing, key=lambda r: r["bits"]) if passing else base
    selection = {"selected": selected, "best_per_width": choices, "profiles": profiles,
                 "test_labels_used_in_selection": False}
    (out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    print("Accumulator selection frozen:", selected, flush=True)
    chosen = sorted({selected["configuration"], "float32", "int16_frac8"})
    test_rows, raw = [], {"labels": data["test_labels"]}
    for model, (w, b) in models.items():
        for name in chosen:
            config = configurations[name]
            layers = [FixedPointLinear(wi, bi, *config) if config else NativeLinear(wi, bi) for wi, bi in zip(w, b)]
            for seed in range(5):
                logits, counts = run(layers, data["test_inputs"], model, seed, protocol["test_domain"], config is not None)
                raw[f"model{model}_seed{seed}_{name}"] = logits
                test_rows.append({"model_seed": model, "sampling_seed": seed, "configuration": name,
                    **metrics(logits, data["test_labels"]),
                    "layer0_increment_clips": int(counts[0, 0]), "layer0_sum_clips": int(counts[0, 1]),
                    "layer1_increment_clips": int(counts[1, 0]), "layer1_sum_clips": int(counts[1, 1])})
            print(f"accumulator test model={model} configuration={name} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", test_rows)
    np.savez_compressed(out/"test_logits.npz", **raw)
    summary = []
    for name in chosen:
        a, delta = [], []
        for model in range(5):
            mean = lambda nm: np.mean([r["accuracy"] for r in test_rows if r["model_seed"] == model and r["configuration"] == nm])
            a.append(mean(name))
            delta.append(mean(name)-mean("float32"))
        group = [r for r in test_rows if r["configuration"] == name]
        summary.append({"configuration": name, "accuracy_mean": float(np.mean(a)),
                        "accuracy_ci95_margin": float(1.96*np.std(a, ddof=1)/np.sqrt(5)),
                        "paired_accuracy_delta_vs_float32": float(np.mean(delta)),
                        "paired_delta_ci95_margin": float(1.96*np.std(delta, ddof=1)/np.sqrt(5)),
                        "total_increment_clips": sum(r["layer0_increment_clips"]+r["layer1_increment_clips"] for r in group),
                        "total_sum_clips": sum(r["layer0_sum_clips"]+r["layer1_sum_clips"] for r in group),
                        "nominal_accumulator_bits_per_example_sequential_peak": 32*profiles[name]["bits"]})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", protocol, {"selection": selection, "summary": summary, "elapsed_seconds": time.monotonic()-start})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
