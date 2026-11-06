"""Uniform comparisons bracketing the actual free-node update budget per model."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .layer_budgets import metrics, run_grid, write_csv
from .positive_inference import NativeLinear
from .provenance import manifest


def main():
    root = Path("results/thrml_spw")
    selection = json.loads((root/"layer_budgets/selection.json").read_text())["selected"]
    t0, t1 = selection["T0"], selection["T1"]
    domain = json.loads((root/"layer_budgets/protocol.json").read_text())["test_key_domain"]
    out = root/"layer_update_matched"
    out.mkdir(parents=True, exist_ok=True)
    layers_by_model, budgets = {}, {}
    for model in range(5):
        z = np.load(f"data/thrml_spw/positive_mnist/seed{model}.npz")
        layers = [NativeLinear(z[f"weight_{i}"], z[f"bias_{i}"]) for i in range(2)]
        free = [len(layer.bank.free_indices) for layer in layers]
        target = free[0]*t0+free[1]*t1
        budgets[str(model)] = {"free_nodes": free, "selected_updates": target,
                              "floor_samples": target//sum(free),
                              "ceil_samples": (target+sum(free)-1)//sum(free)}
        layers_by_model[model] = layers
    rule = {"selected_pair": [t0, t1], "budgets_by_model": budgets,
            "selection_uses_test_labels": False,
            "rule": "Bracket each model's selected free-node update budget by floor/ceiling uniform integer samples. Weights alone determine the bracket; retain both without choosing on test results.",
            "test_key_domain": domain, "sampling_seeds": list(range(5)), "batch_size": 32,
            "scope": "Each retained Gibbs observation costs one free-node update; initial observation is after warmup1, so T samples cost T sweeps. Read counts include clamps; updates exclude them. No chip energy/latency claim."}
    (out/"protocol.json").write_text(json.dumps(rule, indent=2)+"\n")
    data = np.load("data/thrml_spw/positive_mnist/evaluation_data.npz")
    parent = np.load(root/"layer_budgets/test_logits.npz")
    x, y = data["test_inputs"], data["test_labels"]
    raw, rows, start = {"labels": y}, [], time.monotonic()
    for model, layers in layers_by_model.items():
        config = budgets[str(model)]
        pairs = sorted({(config["floor_samples"],)*2, (config["ceil_samples"],)*2})
        for seed in range(5):
            predictions = run_grid(layers, x, pairs, model, seed, domain)
            selected_accuracy = metrics(parent[f"model{model}_seed{seed}_T{t0}_{t1}"], y)["accuracy"]
            for (t, _), logits in predictions.items():
                score = metrics(logits, y)
                raw[f"model{model}_seed{seed}_T{t}"] = logits
                rows.append({"model_seed": model, "sampling_seed": seed,
                             "bracket": "floor" if t == config["floor_samples"] else "ceil",
                             "uniform_samples": t, "logical_reads_per_isolated_example": 2368*t,
                             "free_updates_per_isolated_example": sum(config["free_nodes"])*t,
                             "selected_free_updates_per_isolated_example": config["selected_updates"],
                             "update_difference_from_selected": sum(config["free_nodes"])*t-config["selected_updates"],
                             "selected_minus_uniform_accuracy": selected_accuracy-score["accuracy"], **score})
        print(f"free-update matched model={model} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", rows)
    np.savez_compressed(out/"test_logits.npz", **raw)
    summary = []
    for bracket in ("floor", "ceil"):
        a, d = [], []
        for model in range(5):
            group = [r for r in rows if r["bracket"] == bracket and r["model_seed"] == model]
            a.append(np.mean([r["accuracy"] for r in group]))
            d.append(np.mean([r["selected_minus_uniform_accuracy"] for r in group]))
        summary.append({"bracket": bracket, "uniform_accuracy_mean": float(np.mean(a)),
                        "uniform_accuracy_ci95_margin": float(1.96*np.std(a, ddof=1)/np.sqrt(5)),
                        "selected_minus_uniform_accuracy_mean": float(np.mean(d)),
                        "paired_delta_ci95_margin": float(1.96*np.std(d, ddof=1)/np.sqrt(5))})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", rule, {"summary": summary, "elapsed_seconds": time.monotonic()-start})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
