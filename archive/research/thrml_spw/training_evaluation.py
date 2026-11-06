"""Complete-test paired evaluation of positive-MNIST THRML training exports."""
from __future__ import annotations

import csv
import hashlib
import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from .positive_inference import NativeLinear, deterministic
from .provenance import manifest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score(logits, labels):
    pred = logits.argmax(axis=1)
    return {"accuracy": float(np.mean(pred == labels)),
            "cross_entropy": float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]))}


def run_native(layers, inputs, t, seed, batch_size=32):
    outputs = []
    for batch, start in enumerate(range(0, len(inputs), batch_size)):
        x = jnp.asarray(inputs[start:start + batch_size])
        key = jax.random.fold_in(jax.random.key(seed + 73000), batch)
        k0, k1 = jax.random.split(key)
        hidden = jax.nn.relu(layers[0].compiled(t)(k0, x))
        outputs.append(np.asarray(layers[1].compiled(t)(k1, hidden)))
    return np.concatenate(outputs)


def get_models():
    root = Path("data/thrml_spw/positive_mnist")
    original = np.load(root / "seed0.npz")
    w0 = [original[f"weight_{i}"].astype(np.float32) for i in range(2)]
    b0 = [original[f"bias_{i}"].astype(np.float32) for i in range(2)]
    clipped = [np.clip(w, 1e-4, 1-1e-4).astype(np.float32) for w in w0]
    expected = np.load("results/thrml_spw/native_training/seed0_expected_control_weights.npz")
    native = np.load("results/thrml_spw/native_training/seed0_native_thrml_weights.npz")
    return {
        "original_seed0": (w0, b0),
        "initial_clipped": (clipped, b0),
        "expected_control_2epoch": ([expected[f"weight_{i}"].astype(np.float32) for i in range(2)],
                                    [expected[f"bias_{i}"].astype(np.float32) for i in range(2)]),
        "native_trained_2epoch": ([native[f"weight_{i}"].astype(np.float32) for i in range(2)],
                                  [native[f"bias_{i}"].astype(np.float32) for i in range(2)]),
    }


def quantize_uint4(weights):
    output = []
    for weight in weights:
        scale = float(np.max(weight)) / 15.0
        output.append(np.rint(weight / scale).clip(0, 15) * scale if scale else weight.copy())
    return output


def mean_ci(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    margin = float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return mean, margin


def main():
    out = Path("results/thrml_spw/training_evaluation")
    out.mkdir(parents=True, exist_ok=True)
    eval_path = Path("data/thrml_spw/positive_mnist/evaluation_data.npz")
    evaluation = np.load(eval_path)
    x = evaluation["test_inputs"].astype(np.float32)
    y = evaluation["test_labels"].astype(np.int64)
    if len(y) != 10000:
        raise RuntimeError(f"Expected all 10,000 MNIST test images; found {len(y)}")
    models = get_models()
    sampling_seeds, budgets, batch_size = list(range(5)), (16, 64), 32
    deterministic_logits, deterministic_metrics, controls, layer_banks = {}, {}, {}, {}
    raw = {"labels": y, "test_indices": np.arange(len(y))}
    paired = {"labels": y}
    rows = []
    start_time = time.monotonic()

    for name, (weights, biases) in models.items():
        logits = deterministic(x, weights, biases).astype(np.float32)
        deterministic_logits[name] = logits
        deterministic_metrics[name] = score(logits, y)
        raw[name + "_expected_logits"] = logits
        layer_banks[name] = [NativeLinear(w, b) for w, b in zip(weights, biases)]
        controls[name] = [layer.bank.controls() for layer in layer_banks[name]]
        rows.append({"condition": name, "representation": "deterministic_expected",
                     "sampling_seed": "", "samples_per_layer": 0, "examples": len(y),
                     **deterministic_metrics[name], "accuracy_delta_vs_own_expected": 0.0,
                     "accuracy_delta_vs_original_expected": deterministic_metrics[name]["accuracy"]-
                         deterministic_metrics["original_seed0"]["accuracy"] if "original_seed0" in deterministic_metrics else 0.0,
                     "logit_rmse_vs_own_expected": 0.0,
                     "logical_pbits": sum(w.size for w in weights),
                     "free_pbits": sum(c["free_pbits"] for c in controls[name]),
                     "endpoint_clamped_pbits": sum(c["clamped_endpoint_pbits"] for c in controls[name])})
        qweights = quantize_uint4(weights)
        qlogits = deterministic(x, qweights, biases).astype(np.float32)
        qscore = score(qlogits, y)
        qname = name + "_UINT4_per_layer_scale"
        raw[qname + "_logits"] = qlogits
        rows.append({"condition": name, "representation": "UINT4_per_layer_scale_same_model",
                     "sampling_seed": "", "samples_per_layer": 0, "examples": len(y),
                     **qscore,
                     "accuracy_delta_vs_own_expected": qscore["accuracy"]-deterministic_metrics[name]["accuracy"],
                     "accuracy_delta_vs_original_expected": qscore["accuracy"]-deterministic_metrics["original_seed0"]["accuracy"] if "original_seed0" in deterministic_metrics else 0.0,
                     "logit_rmse_vs_own_expected": float(np.sqrt(np.mean((qlogits-logits)**2))),
                     "logical_pbits": 0, "free_pbits": 0, "endpoint_clamped_pbits": 0})

    # Check export fidelity against the logits created by the training exporter.
    source = np.load("data/thrml_spw/positive_mnist/seed0.npz")
    export_error = float(np.max(np.abs(deterministic_logits["original_seed0"]-
                                      source["reference_test_logits"])))
    if export_error > 1e-4:
        raise RuntimeError(f"seed-0 training/export logit difference too large: {export_error}")
    original_correct = (deterministic_logits["original_seed0"].argmax(1) == y).astype(np.int8)

    # Same complete test rows, sample seed, batch index, and layer-key policy for every
    # representation. The sample-seed interval is conditional on one trained model.
    for name, (weights, biases) in models.items():
        for t in budgets:
            for seed in sampling_seeds:
                logits = run_native(layer_banks[name], x, t, seed, batch_size)
                logits = logits.astype(np.float32)
                result = score(logits, y)
                own = deterministic_logits[name]
                correct = (logits.argmax(1) == y).astype(np.int8)
                delta_own = logits-own
                delta_orig = logits-deterministic_logits["original_seed0"]
                raw[f"{name}_T{t}_seed{seed}_logits"] = logits
                raw[f"{name}_T{t}_seed{seed}_delta_logits_own_expected"] = delta_own
                raw[f"{name}_T{t}_seed{seed}_delta_logits_original_expected"] = delta_orig
                paired[f"{name}_T{t}_seed{seed}_correctness_diff_original"] = correct-original_correct
                rows.append({"condition": name, "representation": "THRML_sequential_preactivation",
                    "sampling_seed": seed, "samples_per_layer": t, "examples": len(y),
                    **result, "accuracy_delta_vs_own_expected": result["accuracy"]-deterministic_metrics[name]["accuracy"],
                    "accuracy_delta_vs_original_expected": result["accuracy"]-deterministic_metrics["original_seed0"]["accuracy"],
                    "paired_correctness_delta_vs_original_expected": float(np.mean(correct-original_correct)),
                    "logit_rmse_vs_own_expected": float(np.sqrt(np.mean(delta_own**2))),
                    "logit_rmse_vs_original_expected": float(np.sqrt(np.mean(delta_orig**2))),
                    "logical_pbits": sum(w.size for w in weights),
                    "free_pbits": sum(c["free_pbits"] for c in controls[name]),
                    "endpoint_clamped_pbits": sum(c["clamped_endpoint_pbits"] for c in controls[name]),
                    "estimated_free_node_gibbs_updates": int(t*((len(y)+batch_size-1)//batch_size)*
                        sum(c["free_pbits"] for c in controls[name]))})
                print(f"{name} seed={seed} T={t} accuracy={result['accuracy']:.4f} "
                      f"det={deterministic_metrics[name]['accuracy']:.4f}", flush=True)

    per_seed_path = out/"metrics_by_sampling_seed.csv"
    with per_seed_path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = []
    metrics = ("accuracy", "cross_entropy", "accuracy_delta_vs_own_expected",
               "accuracy_delta_vs_original_expected", "paired_correctness_delta_vs_original_expected",
               "logit_rmse_vs_own_expected", "logit_rmse_vs_original_expected")
    for name in models:
        for t in budgets:
            group = [r for r in rows if r["condition"] == name and r["representation"] == "THRML_sequential_preactivation" and r["samples_per_layer"] == t]
            row = {"condition": name, "samples_per_layer": t, "sampling_repeats": len(group),
                   "training_model_seeds": 1,
                   "deterministic_expected_accuracy": deterministic_metrics[name]["accuracy"],
                   "deterministic_expected_cross_entropy": deterministic_metrics[name]["cross_entropy"]}
            for metric in metrics:
                avg, margin = mean_ci([r[metric] for r in group])
                row[metric+"_mean"] = avg
                row[metric+"_sampling_seed_95ci_half_width"] = margin
            summary.append(row)
    with (out/"sampling_seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    np.savez_compressed(out/"raw_logits.npz", **raw)
    np.savez_compressed(out/"paired_differences.npz", **paired)
    details = {
        "elapsed_seconds": time.monotonic()-start_time, "test_examples": len(y),
        "sampling_seeds": sampling_seeds, "T_values": list(budgets), "batch_size": batch_size,
        "key_policy": "fold_in(key(sampling_seed + 73000), batch_index), then split by layer; shared across model conditions",
        "execution": "positive_inference.NativeLinear: THRML sample_with_observation per layer; average preactivation; ReLU once; independent second-layer sampler",
        "deterministic_biases": True, "test_set_used_for_selection": False,
        "sampling_CI_scope": "normal-approximation 95% interval per requested convention, computed as mean +/- 1.96*s/sqrt(5) across repeated sampling seeds conditional on the fixed seed-0 model; no training-model variation CI (only one trained checkpoint)",
        "deterministic_expected_metrics": deterministic_metrics,
        "sampler_controls": controls,
        "endpoint_shape_note": "original has exact zero/one endpoint nodes clamped by NativeProbabilityBank; initial-clipped and both trained exports have all weight nodes free in the open unit interval",
        "source_checkpoint_logit_max_error": export_error,
        "input_sha256": sha256(eval_path),
        "expected_export_sha256": sha256("results/thrml_spw/native_training/seed0_expected_control_weights.npz"),
        "native_export_sha256": sha256("results/thrml_spw/native_training/seed0_native_thrml_weights.npz"),
    }
    (out/"details.json").write_text(json.dumps(details, indent=2)+"\n")
    manifest(out/"manifest.json", {"test_examples": len(y), "sampling_seeds": sampling_seeds,
             "T_values": list(budgets), "batch_size": batch_size, "conditions": list(models),
             "same_model_UINT4_per_layer": True}, details)
    print(f"saved full-test repeated-sampling evidence to {out}", flush=True)


if __name__ == "__main__":
    main()
