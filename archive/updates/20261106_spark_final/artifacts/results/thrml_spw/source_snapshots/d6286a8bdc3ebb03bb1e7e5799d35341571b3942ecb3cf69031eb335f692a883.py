"""Actual shared-bank online validation and five-by-five test confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .adaptive_shared import SharedAdaptiveNetwork
from .adaptive_study import ci, gamma_value
from .layer_budgets import metrics, write_csv
from .provenance import manifest


def main():
    root, source = Path("results/thrml_spw"), Path("data/thrml_spw/positive_mnist")
    out = root/"adaptive_shared"
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(source/"evaluation_data.npz")
    gammas = ["0", "1", "2", "3", "4", "6", "8", "12", "16", "never"]
    val_domain, test_domain, batch = 840730, 850730, 128
    protocol = {"trained_model_seeds": list(range(5)), "validation_sampling_seeds": [0],
                "test_sampling_seeds": list(range(5)), "gamma_candidates": gammas,
                "first_layer_samples": 64, "output_checkpoints": [16, 32, 64, 128, 256],
                "validation_examples": 5000, "test_examples": 10000, "batch_size": batch,
                "validation_domain": val_domain, "test_domain": test_domain,
                "test_labels_used_in_selection": False,
                "selection_rule": "Minimum mean executed free-node updates on actual online shared-bank validation, with accuracy at least full256 minus0.002 and CE at most full256 plus0.02. Ties by CE and gamma order.",
                "execution": "One native THRML bank state broadcast across up to128 examples. Compact active cases between chunks. Reset uncoupled bank with warmup1; one new draw per observation.",
                "path_note": "Compaction changes later batch membership/keys; fixed-reference paths generally differ after stopping. Stored online decision traces, not a different reference replay, establish policy consistency.",
                "resource_accounting": "Count actual bank requests times samples times free nodes; padding adds rows and may add a final bank request, not128 independent bank chains. Also report padded dot-product term count.",
                "comparator": "Per trained model and sampling seed, bracket online second-layer bank-observation budget with uniform integer T; use counters only, no labels. First-layer cost common to every condition.",
                "confidence_scope": "Conditional output-layer margin/empirical-SE heuristic; excludes first-layer uncertainty and provides no anytime coverage guarantee",
                "uncertainty": "Normal-approximation 1.96*SD/sqrt(5) across trained-model means after averaging five sampling seeds; paired differences across model means",
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [source/"evaluation_data.npz"]+[source/f"seed{s}.npz" for s in range(5)]},
                "scope": "Dimensionless THRML CPU/JAX model with float32 external arithmetic; no physical fanout, energy or latency measurement"}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    models, val_rows, val_raw, val_counters = {}, [], {"labels": data["validation_labels"]}, {}
    start = time.monotonic()
    for model in range(5):
        z = np.load(source/f"seed{model}.npz")
        net = SharedAdaptiveNetwork([z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)], batch)
        models[model] = net
        hidden, first_obs = net.hidden(data["validation_inputs"], val_domain, model, 0)
        f0, f1 = len(net.first.bank.free_indices), len(net.bank.free_indices)
        for gamma in gammas:
            logits, stop, counts, history, moments = net.online(hidden, val_domain, model, 0, gamma_value(gamma))
            second_obs = sum(c["bank_observations"] for c in counts)
            val_raw[f"model{model}_gamma{gamma}_logits"] = logits
            val_raw[f"model{model}_gamma{gamma}_stop"] = stop
            val_counters[f"model{model}_gamma{gamma}"] = counts
            val_rows.append({"model_seed": model, "gamma": gamma,
                **metrics(logits, data["validation_labels"]), "mean_output_samples": float(stop.mean()),
                "first_bank_observations": first_obs, "second_bank_observations": second_obs,
                "executed_free_node_updates": first_obs*f0+second_obs*f1})
        print(f"shared adaptive validation model={model} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"validation_metrics.csv", val_rows)
    np.savez_compressed(out/"validation_outputs.npz", **val_raw)
    (out/"validation_counters.json").write_text(json.dumps(val_counters, indent=2)+"\n")
    profiles = []
    for gamma in gammas:
        group = [r for r in val_rows if r["gamma"] == gamma]
        profiles.append({"gamma": gamma, "accuracy_mean": float(np.mean([r["accuracy"] for r in group])),
                         "cross_entropy_mean": float(np.mean([r["cross_entropy"] for r in group])),
                         "mean_output_samples": float(np.mean([r["mean_output_samples"] for r in group])),
                         "executed_free_updates_mean": float(np.mean([r["executed_free_node_updates"] for r in group]))})
    ref = next(r for r in profiles if r["gamma"] == "never")
    passing = [r for r in profiles if r["accuracy_mean"] >= ref["accuracy_mean"]-.002 and r["cross_entropy_mean"] <= ref["cross_entropy_mean"]+.02]
    selected = min(passing, key=lambda r: (r["executed_free_updates_mean"], r["cross_entropy_mean"], gammas.index(r["gamma"])))
    selection = {"selected": selected, "reference": ref, "profiles": profiles, "test_labels_used_in_selection": False}
    (out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    print("Shared rule frozen:", selected, flush=True)
    x, labels = data["test_inputs"], data["test_labels"]
    raw, rows, counters = {"labels": labels}, [], {}
    for model, net in models.items():
        f0, f1 = len(net.first.bank.free_indices), len(net.bank.free_indices)
        for seed in range(5):
            hidden, first_obs = net.hidden(x, test_domain, model, seed)
            fixed_history, _, fixed_counts = net.collect(hidden, test_domain, model, seed)
            online_hidden, _ = net.hidden(x, test_domain, model, seed)
            np.testing.assert_array_equal(hidden, online_hidden)
            online, stop, counts, history, moments = net.online(online_hidden, test_domain, model, seed, gamma_value(selected["gamma"]))
            second_obs = sum(c["bank_observations"] for c in counts)
            bank_batches = (len(x)+batch-1)//batch
            outputs = {"full256": (fixed_history[-1], sum(c["bank_observations"] for c in fixed_counts), 256.),
                       "adaptive": (online, second_obs, float(stop.mean()))}
            for bracket, t in (("matched_floor", second_obs//bank_batches), ("matched_ceil", (second_obs+bank_batches-1)//bank_batches)):
                trace, _, costs = net.collect(hidden, test_domain, model, seed, t)
                outputs[bracket] = (trace[-1], sum(c["bank_observations"] for c in costs), float(t))
            key = f"model{model}_seed{seed}"
            raw[key+"_stop"], raw[key+"_online_history"], raw[key+"_online_m2"] = stop, history, moments
            counters[key] = {"online": counts, "reference": fixed_counts, "free_nodes": [f0, f1], "first_bank_observations": first_obs}
            reference_accuracy = metrics(fixed_history[-1], labels)["accuracy"]
            for name, (logits, observations, mean_t) in outputs.items():
                raw[key+"_"+name] = logits
                score = metrics(logits, labels)
                rows.append({"model_seed": model, "sampling_seed": seed, "condition": name, **score,
                    "accuracy_delta_vs_full256": score["accuracy"]-reference_accuracy,
                    "mean_output_samples": mean_t, "first_bank_observations": first_obs,
                    "second_bank_observations": observations,
                    "executed_free_node_updates": first_obs*f0+observations*f1,
                    "logical_weight_readouts": first_obs*2048+observations*320,
                    "padded_binary_product_terms": batch*(first_obs*2048+observations*320)})
            print(f"shared adaptive test model={model} seed={seed} meanT={stop.mean():.2f} accuracy={metrics(online,labels)['accuracy']:.4f} full={reference_accuracy:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", rows)
    np.savez_compressed(out/"test_outputs.npz", **raw)
    (out/"test_counters.json").write_text(json.dumps(counters, indent=2)+"\n")
    summary = []
    for condition in ("full256", "adaptive", "matched_floor", "matched_ceil"):
        a, d, paired, fractions, costs, sample_means = [], [], [], [], [], []
        for model in range(5):
            group = [r for r in rows if r["model_seed"] == model and r["condition"] == condition]
            a.append(np.mean([r["accuracy"] for r in group]))
            d.append(np.mean([r["accuracy_delta_vs_full256"] for r in group]))
            paired.append(np.mean([next(s["accuracy"] for s in rows if s["model_seed"] == model and s["sampling_seed"] == r["sampling_seed"] and s["condition"] == "adaptive")-r["accuracy"] for r in group]))
            fractions.extend([r["executed_free_node_updates"]/next(s["executed_free_node_updates"] for s in rows if s["model_seed"] == model and s["sampling_seed"] == r["sampling_seed"] and s["condition"] == "full256") for r in group])
            costs.extend(r["executed_free_node_updates"] for r in group)
            sample_means.extend(r["mean_output_samples"] for r in group)
        summary.append({"condition": condition, "accuracy_mean": float(np.mean(a)), "accuracy_ci95_margin": ci(a),
                        "paired_accuracy_delta_vs_full256": float(np.mean(d)), "paired_delta_ci95_margin": ci(d),
                        "adaptive_minus_condition_accuracy_mean": float(np.mean(paired)), "adaptive_minus_condition_ci95_margin": ci(paired),
                        "mean_fraction_of_full256_updates": float(np.mean(fractions)),
                        "mean_executed_free_node_updates": float(np.mean(costs)), "mean_output_samples": float(np.mean(sample_means))})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", protocol, {"selection": selection, "summary": summary,
             "elapsed_seconds": time.monotonic()-start,
             "arithmetic_scope": "Batch-shared float32 moment observer; actual compaction changes reference paths. No claim of unbiased stopped outputs or hardware advantage."})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
