"""Frozen larger-network generalization using genuine THRML linear observers.

Three explicitly separate routes: direct p=w on positive weights; per-output
range encoding of the same weights; and range encoding of signed control weights.
Validation alone chooses layer budgets against a deterministic quality target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .layer_budgets import metrics, run_grid, write_csv
from .positive_inference import NativeLinear, deterministic
from .provenance import manifest
from .range_native import RangeNativeLinear


ROUTES = {"positive_direct": "positive_projected", "positive_range": "positive_projected",
          "signed_range": "signed_released_control"}


def quantized(weights, bits, grouping):
    result = []
    for w in weights:
        if grouping == "per_output":
            lo, hi = w.min(0), w.max(0)
        else:
            lo, hi = float(w.min()), float(w.max())
        span = hi-lo
        p = np.zeros_like(w)
        np.divide(w-lo, span, out=p, where=span != 0)
        result.append((lo+span*np.rint(p*((1 << bits)-1))/((1 << bits)-1)).astype(np.float32))
    return result


def info(layers):
    return [layer.controls() if isinstance(layer, RangeNativeLinear) else layer.bank.controls() for layer in layers]


def ci(x):
    return float(1.96*np.std(x, ddof=1)/np.sqrt(len(x)))


def main():
    source = Path("data/thrml_spw/positive_fashion")
    training = json.loads(Path("results/thrml_spw/fashion_training/manifest.json").read_text())
    assert len(training["models"]) == 10
    data = np.load(source/"evaluation_data.npz")
    out = Path("results/thrml_spw/fashion_native")
    out.mkdir(parents=True, exist_ok=True)
    values = [16, 64, 256, 1024]
    pairs = [(a, b) for a in values for b in values]
    protocol = {"routes": ROUTES, "matrix_shapes": [[196, 64], [64, 10]], "matrix_weights": 13184,
                "deterministic_biases": 74, "grouping": "per output; affine routes add2 float32 metadata values per output column",
                "layer_sample_values": values, "validation_examples": 5000, "test_examples": 10000,
                "trained_model_seeds": list(range(5)), "validation_sampling_seeds": [0],
                "test_sampling_seeds": list(range(5)), "batch_size": 128,
                "validation_domain_by_route": {r: 910730+i for i, r in enumerate(ROUTES)},
                "test_domain_by_route": {r: 920730+i for i, r in enumerate(ROUTES)},
                "selection_rule": "For each route, minimize mean free-node updates per isolated inference with mean validation accuracy within0.01 and CE within0.05 of that trained model's deterministic baseline. Ties by CE then budgets. If no candidate passes, record failure and test lowest-CE candidate without redefining the gate.",
                "test_labels_used_in_selection": False,
                "test_comparisons": "Selected allocation, uniform256 and per-model uniform ceiling of selected free-update budget. Budget ceiling uses weight/control counts only, no test labels.",
                "uncertainty": "1.96*SD/sqrt(5) across trained-model means after averaging five test sampling seeds; paired model-mean accuracy differences",
                "resource_scope": "Shared bank states across128 inputs. One native update per free node per observation; endpoints clamped. Actual evaluated dataset counts multiply by number of bank batches, isolated counts do not.",
                "arithmetic_scope": "Native THRML Gibbs states; float32 external JAX dot products, sums, affine scaling and activation. No physical timing/energy claim.",
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [source/"evaluation_data.npz", Path("results/thrml_spw/fashion_training/manifest.json")]+sorted(source.glob("seed*.npz"))}}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    models, val_rows, val_raw, baseline_val, controls, variance = {}, [], {"labels": data["validation_labels"]}, {}, {}, []
    start, max_export_error = time.monotonic(), 0.
    for model in range(5):
        for route, condition in ROUTES.items():
            z = np.load(source/f"seed{model}_{condition}.npz")
            weights, biases = [z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)]
            reference = deterministic(data["validation_inputs"], weights, biases)
            error = float(np.max(np.abs(reference-z["reference_validation_logits"])))
            max_export_error = max(max_export_error, error)
            if error > 1e-4:
                raise RuntimeError(f"Training/export mismatch: {error}")
            baseline_val[(model, route)] = metrics(reference, data["validation_labels"])
            cls = NativeLinear if route == "positive_direct" else RangeNativeLinear
            layers = [cls(w, b) for w, b in zip(weights, biases)]
            models[(model, route)] = (weights, biases, layers)
            controls[f"model{model}_{route}"] = info(layers)
            free = [len(layer.bank.free_indices) for layer in layers]
            for i, (w, layer) in enumerate(zip(weights, layers)):
                q = layer.bank.realized_probabilities
                sample_var = q*(1-q)
                if isinstance(layer, RangeNativeLinear):
                    sample_var = sample_var*layer.span**2
                    weight_mean = layer.realized_weights
                else:
                    weight_mean = q
                variance.append({"model_seed": model, "route": route, "layer": i,
                                 "mean_single_observation_weight_variance": float(sample_var.mean()),
                                 "max_expected_weight_rounding_error": float(np.max(np.abs(weight_mean-w)))})
            predictions = run_grid(layers, data["validation_inputs"], pairs, model, 0,
                                   protocol["validation_domain_by_route"][route], 128)
            for (t0, t1), logits in predictions.items():
                val_raw[f"model{model}_{route}_T{t0}_{t1}"] = logits
                val_rows.append({"model_seed": model, "route": route, "T0": t0, "T1": t1,
                    **metrics(logits, data["validation_labels"]),
                    "FP32_accuracy": baseline_val[(model, route)]["accuracy"],
                    "FP32_cross_entropy": baseline_val[(model, route)]["cross_entropy"],
                    "free_updates_per_isolated_example": free[0]*t0+free[1]*t1,
                    "logical_readouts_per_isolated_example": weights[0].size*t0+weights[1].size*t1})
            print(f"fashion validation model={model} route={route} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"validation_metrics.csv", val_rows)
    write_csv(out/"weight_variance.csv", variance)
    np.savez_compressed(out/"validation_logits.npz", **val_raw)
    selections = {}
    for route in ROUTES:
        profiles = []
        for t0, t1 in pairs:
            group = [r for r in val_rows if r["route"] == route and r["T0"] == t0 and r["T1"] == t1]
            profiles.append({"T0": t0, "T1": t1, "accuracy_mean": float(np.mean([r["accuracy"] for r in group])),
                "cross_entropy_mean": float(np.mean([r["cross_entropy"] for r in group])),
                "FP32_accuracy_mean": float(np.mean([r["FP32_accuracy"] for r in group])),
                "FP32_cross_entropy_mean": float(np.mean([r["FP32_cross_entropy"] for r in group])),
                "free_updates_mean": float(np.mean([r["free_updates_per_isolated_example"] for r in group]))})
        passing = [r for r in profiles if r["accuracy_mean"] >= r["FP32_accuracy_mean"]-.01
                   and r["cross_entropy_mean"] <= r["FP32_cross_entropy_mean"]+.05]
        selected = min(passing, key=lambda r: (r["free_updates_mean"], r["cross_entropy_mean"], r["T0"], r["T1"])) if passing else min(profiles, key=lambda r: r["cross_entropy_mean"])
        selections[route] = {"gate_passed_on_validation": bool(passing), "selected": selected, "profiles": profiles}
        print("Fashion selection frozen:", route, selections[route]["gate_passed_on_validation"], selected, flush=True)
    (out/"selection.json").write_text(json.dumps({"test_labels_used_in_selection": False, "routes": selections}, indent=2)+"\n")
    test_raw, test_rows = {"labels": data["test_labels"]}, []
    for (model, route), (weights, biases, layers) in models.items():
        x, labels = data["test_inputs"], data["test_labels"]
        reference = deterministic(x, weights, biases)
        z = np.load(source/f"seed{model}_{ROUTES[route]}.npz")
        error = float(np.max(np.abs(reference-z["reference_test_logits"])))
        max_export_error = max(max_export_error, error)
        assert error < 1e-4
        fp_accuracy = metrics(reference, labels)["accuracy"]
        def record(condition, logits, sampling_seed=-1, t0=0, t1=0):
            free = [len(layer.bank.free_indices) for layer in layers]
            updates = free[0]*t0+free[1]*t1
            reads = weights[0].size*t0+weights[1].size*t1
            score = metrics(logits, labels)
            test_raw[f"model{model}_{route}_{condition}_seed{sampling_seed}"] = logits
            test_rows.append({"model_seed": model, "route": route, "condition": condition,
                "sampling_seed": sampling_seed, "T0": t0, "T1": t1, **score,
                "accuracy_delta_vs_FP32": score["accuracy"]-fp_accuracy,
                "free_updates_per_isolated_example": updates,
                "dataset_free_node_updates": ((len(x)+127)//128)*updates,
                "logical_readouts_per_isolated_example": reads,
                "metadata_bits": sum(c.get("metadata_bits", 0) for c in controls[f"model{model}_{route}"])})
        record("FP32", reference)
        record("FP16_rounded_values", deterministic(x, [w.astype(np.float16).astype(np.float32) for w in weights], [b.astype(np.float16).astype(np.float32) for b in biases]))
        for bits, grouping in ((4, "per_layer"), (4, "per_output"), (8, "per_output")):
            record(f"affine{bits}_{grouping}", deterministic(x, quantized(weights, bits, grouping), biases))
        chosen = selections[route]["selected"]
        t0, t1 = chosen["T0"], chosen["T1"]
        free = [len(layer.bank.free_indices) for layer in layers]
        matched = (free[0]*t0+free[1]*t1+sum(free)-1)//sum(free)
        conditions = {"selected_native": (t0, t1), "uniform256_native": (256, 256),
                      "matched_updates_ceil_native": (matched, matched)}
        for seed in range(5):
            predictions = run_grid(layers, x, sorted(set(conditions.values())), model, seed,
                                   protocol["test_domain_by_route"][route], 128)
            for name, pair in conditions.items():
                record(name, predictions[pair], seed, *pair)
        print(f"fashion test model={model} route={route} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", test_rows)
    np.savez_compressed(out/"test_logits.npz", **test_raw)
    summary = []
    for route in ROUTES:
        for condition in sorted({r["condition"] for r in test_rows if r["route"] == route}):
            group = [r for r in test_rows if r["route"] == route and r["condition"] == condition]
            a, d, paired = [], [], []
            for model in range(5):
                rows = [r for r in group if r["model_seed"] == model]
                a.append(np.mean([r["accuracy"] for r in rows]))
                d.append(np.mean([r["accuracy_delta_vs_FP32"] for r in rows]))
                matched_rows = [r for r in test_rows if r["model_seed"] == model and r["route"] == route and r["condition"] == "matched_updates_ceil_native"]
                paired.append(a[-1]-np.mean([r["accuracy"] for r in matched_rows]))
            summary.append({"route": route, "condition": condition, "accuracy_mean": float(np.mean(a)),
                "accuracy_ci95_margin": ci(a), "paired_accuracy_delta_vs_FP32": float(np.mean(d)), "paired_delta_ci95_margin": ci(d),
                "paired_accuracy_delta_vs_matched_ceil": float(np.mean(paired)), "paired_matched_ci95_margin": ci(paired),
                "free_updates_per_isolated_example_mean": float(np.mean([r["free_updates_per_isolated_example"] for r in group])),
                "logical_readouts_per_isolated_example_mean": float(np.mean([r["logical_readouts_per_isolated_example"] for r in group]))})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", protocol, {"selections": selections, "summary": summary, "controls": controls,
             "max_torch_numpy_export_logit_error": max_export_error, "elapsed_seconds": time.monotonic()-start,
             "scope": "Larger frozen-network simulation with direct and affine encodings explicitly separated; no hardware or software-memory compression claim."})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
