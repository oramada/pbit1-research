"""Bracket the selected nonuniform budget with exact integer uniform budgets.

The original sweep's uniform grid is coarse. This independent follow-up prevents
an apparent advantage arising just from unused budget between grid points.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .layer_budgets import metrics, run_grid, write_csv
from .positive_inference import NativeLinear
from .provenance import manifest


def main():
    parent = Path("results/thrml_spw/layer_budgets")
    selection = json.loads((parent/"selection.json").read_text())
    protocol = json.loads((parent/"protocol.json").read_text())
    chosen = selection["selected"]
    cost = chosen["logical_reads_per_example"]
    floor, ceil = cost//2368, (cost+2367)//2368
    pairs = sorted({(floor, floor), (ceil, ceil)})
    out = Path("results/thrml_spw/layer_budget_matched")
    out.mkdir(parents=True, exist_ok=True)
    rule = {"selected_pair": [chosen["T0"], chosen["T1"]],
            "selected_logical_reads_per_example": cost,
            "uniform_pairs": pairs, "selection_uses_test_labels": False,
            "rule": "Evaluate floor and ceiling of selected logical-read budget divided by 2368 weights. Use original independent test keys and five trained models by five sampling seeds. No further allocation selection.",
            "parent_selection_sha256": hashlib.sha256((parent/"selection.json").read_bytes()).hexdigest(),
            "keys": {"domain": protocol["test_key_domain"], "fold_order": ["model_seed", "sampling_seed", "batch_index"]},
            "counter_scope": "Logical sampled reads incl. clamps; free updates and serial layer depth also differ. No energy/latency equality is claimed."}
    (out/"protocol.json").write_text(json.dumps(rule, indent=2)+"\n")
    data = np.load("data/thrml_spw/positive_mnist/evaluation_data.npz")
    selected_raw = np.load(parent/"test_logits.npz")
    x, y = data["test_inputs"], data["test_labels"]
    rows, raw, start = [], {"labels": y}, time.monotonic()
    for model in range(5):
        z = np.load(f"data/thrml_spw/positive_mnist/seed{model}.npz")
        layers = [NativeLinear(z[f"weight_{i}"], z[f"bias_{i}"]) for i in range(2)]
        for seed in range(5):
            outputs = run_grid(layers, x, pairs, model, seed, protocol["test_key_domain"])
            selected_logits = selected_raw[f"model{model}_seed{seed}_T{chosen['T0']}_{chosen['T1']}"]
            selected_accuracy = metrics(selected_logits, y)["accuracy"]
            for (a, b), logits in outputs.items():
                score = metrics(logits, y)
                free = [len(layer.bank.free_indices) for layer in layers]
                selected_updates = free[0]*chosen["T0"]+free[1]*chosen["T1"]
                rows.append({"model_seed": model, "sampling_seed": seed,
                             "uniform_samples": a, "logical_reads_per_example": 2368*a,
                             "uniform_read_difference_from_selected": 2368*a-cost,
                             "dependent_layer_sweeps": a+b,
                             "free_updates_per_isolated_example": sum(free)*a,
                             "selected_free_updates_per_isolated_example": selected_updates,
                             "selected_accuracy": selected_accuracy,
                             "selected_accuracy_minus_uniform": selected_accuracy-score["accuracy"], **score})
                raw[f"model{model}_seed{seed}_T{a}"] = logits
        print(f"exact-budget uniform comparison model={model} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", rows)
    np.savez_compressed(out/"test_logits.npz", **raw)
    summary = []
    for t, _ in pairs:
        a, d = [], []
        for model in range(5):
            group = [r for r in rows if r["uniform_samples"] == t and r["model_seed"] == model]
            a.append(np.mean([r["accuracy"] for r in group]))
            d.append(np.mean([r["selected_accuracy_minus_uniform"] for r in group]))
        summary.append({"uniform_samples": t, "logical_reads_per_example": 2368*t,
                        "uniform_accuracy_mean": float(np.mean(a)),
                        "uniform_accuracy_ci95_margin": float(1.96*np.std(a, ddof=1)/np.sqrt(5)),
                        "selected_minus_uniform_mean": float(np.mean(d)),
                        "paired_delta_ci95_margin": float(1.96*np.std(d, ddof=1)/np.sqrt(5))})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", rule, {"summary": summary, "elapsed_seconds": time.monotonic()-start})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
