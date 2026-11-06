"""Validation-selected layer sample allocation using actual THRML inference.

No test labels enter selection. Each matrix's sampled linear outputs are averaged
before its nonlinearity. All arithmetic remains a CPU/JAX observer model.
"""
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


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(logits, labels):
    return {"accuracy": float(np.mean(logits.argmax(1) == labels)),
            "cross_entropy": float(np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(labels)), labels]))}


def run_grid(layers, x, budgets, model_seed, sampling_seed, domain, batch_size=32):
    """Reuse first-layer outputs across second-layer choices on the same keys.

    This accelerates the grid study; reported deployment costs describe a single
    chosen pair executing without cross-candidate reuse.
    """
    outputs = {pair: [] for pair in budgets}
    # Keep local references: NativeLinear's shared LRU holds only eight entries,
    # while this grid needs ten layer/budget functions. Looking them up inside
    # every batch would repeatedly evict and recompile otherwise identical code.
    first_layer = {t: layers[0].compiled(t) for t in sorted({a for a, _ in budgets})}
    second_layer = {t: layers[1].compiled(t) for t in sorted({b for _, b in budgets})}
    for batch, first in enumerate(range(0, len(x), batch_size)):
        key = jax.random.key(domain)
        for value in (model_seed, sampling_seed, batch):
            key = jax.random.fold_in(key, value)
        k0, k1 = jax.random.split(key)
        xb = jnp.asarray(x[first:first+batch_size])
        hidden = {t: jax.nn.relu(fn(k0, xb)) for t, fn in first_layer.items()}
        for t0, t1 in budgets:
            outputs[(t0, t1)].append(np.asarray(second_layer[t1](k1, hidden[t0])))
    return {pair: np.concatenate(value) for pair, value in outputs.items()}


def main():
    out = Path("results/thrml_spw/layer_budgets")
    out.mkdir(parents=True, exist_ok=True)
    source = Path("data/thrml_spw/positive_mnist")
    data = np.load(source/"evaluation_data.npz")
    xv, yv = data["validation_inputs"], data["validation_labels"]
    values = [16, 32, 64, 128, 256]
    pairs = [(a, b) for a in values for b in values]
    model_seeds, validation_seeds, test_seeds = list(range(5)), list(range(2)), list(range(5))
    n0, n1 = 2048, 320
    protocol = {
        "layer_sample_values": values, "candidate_pairs": pairs,
        "models": model_seeds, "validation_sampling_seeds": validation_seeds,
        "test_sampling_seeds": test_seeds, "validation_examples": 5000,
        "test_examples": 10000, "batch_size": 32,
        "selection": "Cheapest logical weight-read budget with mean validation accuracy within 0.005 and CE at most 0.02 above uniform (256,256); ties by CE then layer sample counts.",
        "cost": "2048*T0+320*T1 logical weight reads per isolated example, including clamped endpoints",
        "validation_key_domain": 200730, "test_key_domain": 400730,
        "selection_uses_test_labels": False,
        "interval": "1.96*SD/sqrt(5) across trained-model means after averaging five sampling seeds; paired differences use paired model means",
        "study_grid_reuse": "First-layer outputs reused across second-layer candidates; deployment counts exclude cross-candidate reuse.",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [source/"evaluation_data.npz"]+[source/f"seed{s}.npz" for s in model_seeds]},
        "arithmetic_scope": "External CPU/JAX observer arithmetic; no physical timing or energy inference",
    }
    # Save the rule and source hashes before running the validation experiment.
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    models, layers_by_seed, rows, raw = {}, {}, [], {"labels": yv}
    start = time.monotonic()
    for model in model_seeds:
        z = np.load(source/f"seed{model}.npz")
        w, b = [z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)]
        models[model] = (w, b)
        layers = [NativeLinear(wi, bi) for wi, bi in zip(w, b)]
        layers_by_seed[model] = layers
        for seed in validation_seeds:
            predictions = run_grid(layers, xv, pairs, model, seed, protocol["validation_key_domain"])
            for (t0, t1), logits in predictions.items():
                rows.append({"model_seed": model, "sampling_seed": seed, "T0": t0, "T1": t1,
                             "logical_reads_per_example": n0*t0+n1*t1, **metrics(logits, yv)})
                raw[f"model{model}_seed{seed}_T{t0}_{t1}"] = logits
            print(f"validation model={model} sampling_seed={seed} pairs={len(pairs)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"validation_metrics.csv", rows)
    np.savez_compressed(out/"validation_logits.npz", **raw)
    profiles = []
    for t0, t1 in pairs:
        group = [r for r in rows if r["T0"] == t0 and r["T1"] == t1]
        profiles.append({"T0": t0, "T1": t1, "logical_reads_per_example": n0*t0+n1*t1,
                         "accuracy_mean": float(np.mean([r["accuracy"] for r in group])),
                         "cross_entropy_mean": float(np.mean([r["cross_entropy"] for r in group]))})
    reference = next(r for r in profiles if r["T0"] == 256 and r["T1"] == 256)
    passing = [r for r in profiles if r["accuracy_mean"] >= reference["accuracy_mean"]-.005
               and r["cross_entropy_mean"] <= reference["cross_entropy_mean"]+.02]
    selected = min(passing, key=lambda r: (r["logical_reads_per_example"], r["cross_entropy_mean"], r["T0"], r["T1"]))
    equal_cost_uniform = max(t for t in values if (n0+n1)*t <= selected["logical_reads_per_example"])
    selection = {"selected": selected, "reference": reference,
                 "uniform_within_selected_read_budget": equal_cost_uniform,
                 "test_labels_used_in_selection": False, "profiles": profiles}
    (out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    print("Selection frozen:", selected, "uniform budget comparator", equal_cost_uniform, flush=True)
    chosen = sorted({(selected["T0"], selected["T1"]), (256, 256), (equal_cost_uniform, equal_cost_uniform), (64, 64)})
    x, y = data["test_inputs"], data["test_labels"]
    test_raw, test_rows = {"labels": y}, []
    for model in model_seeds:
        layers = layers_by_seed[model]
        w, b = models[model]
        expected = deterministic(x, w, b)
        test_raw[f"model{model}_expected_logits"] = expected
        fp = metrics(expected, y)
        free = [len(layer.bank.free_indices) for layer in layers]
        for seed in test_seeds:
            predictions = run_grid(layers, x, chosen, model, seed, protocol["test_key_domain"])
            for (t0, t1), logits in predictions.items():
                test_raw[f"model{model}_seed{seed}_T{t0}_{t1}"] = logits
                test_rows.append({"model_seed": model, "sampling_seed": seed, "T0": t0, "T1": t1,
                    "logical_reads_per_isolated_example": n0*t0+n1*t1,
                    "free_node_updates_per_isolated_example": free[0]*t0+free[1]*t1,
                    "dependent_layer_sweeps": t0+t1,
                    "actual_dataset_free_node_updates": ((len(y)+31)//32)*(free[0]*t0+free[1]*t1),
                    "FP32_accuracy": fp["accuracy"], **metrics(logits, y)})
            print(f"test model={model} sampling_seed={seed} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", test_rows)
    np.savez_compressed(out/"test_logits.npz", **test_raw)
    summary = []
    for t0, t1 in chosen:
        a, paired, equal_paired = [], [], []
        for model in model_seeds:
            mean_for = lambda pair: np.mean([r["accuracy"] for r in test_rows if r["model_seed"] == model and (r["T0"], r["T1"]) == pair])
            value = mean_for((t0, t1))
            a.append(value)
            paired.append(value-mean_for((256, 256)))
            equal_paired.append(value-mean_for((equal_cost_uniform, equal_cost_uniform)))
        ci = lambda vals: float(1.96*np.std(vals, ddof=1)/np.sqrt(len(vals)))
        summary.append({"T0": t0, "T1": t1, "trained_models": 5, "sampling_seeds_per_model": 5,
            "accuracy_mean": float(np.mean(a)), "accuracy_ci95_margin": ci(a),
            "paired_accuracy_delta_vs_uniform256": float(np.mean(paired)), "paired_delta_ci95_margin": ci(paired),
            "paired_accuracy_delta_vs_uniform_budget": float(np.mean(equal_paired)), "paired_budget_delta_ci95_margin": ci(equal_paired),
            "logical_reads_per_isolated_example": n0*t0+n1*t1,
            "reads_fraction_of_uniform256": (n0*t0+n1*t1)/((n0+n1)*256),
            "dependent_layer_sweeps": t0+t1})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", protocol, {"selection": selection, "summary": summary,
             "elapsed_seconds": time.monotonic()-start,
             "controls_by_model": {str(s): [l.bank.controls() for l in layers_by_seed[s]] for s in model_seeds}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
