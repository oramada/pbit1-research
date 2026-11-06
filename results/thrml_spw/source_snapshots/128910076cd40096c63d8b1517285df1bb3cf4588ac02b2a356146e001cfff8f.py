"""Calibrate a margin/SE rule, then confirm genuine online THRML stopping.

Validation traces are full-horizon sampling used only to choose a rule. The test
adaptive branch invokes the sampler only for unfinished cases; raw replay checks
show that compaction preserves each retained trajectory. Padding is charged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .adaptive_native import AdaptiveNetwork, CHECKPOINTS, compact_observations, replay
from .layer_budgets import metrics, write_csv
from .positive_inference import deterministic
from .provenance import manifest


def gamma_value(name):
    return np.inf if name == "never" else float(name)


def ci(values):
    return float(1.96*np.std(values, ddof=1)/np.sqrt(len(values)))


def main():
    out = Path("results/thrml_spw/adaptive_native")
    out.mkdir(parents=True, exist_ok=True)
    source = Path("data/thrml_spw/positive_mnist")
    data = np.load(source/"evaluation_data.npz")
    gamma_names = ["0", "1", "2", "3", "4", "6", "8", "12", "16", "never"]
    model_seeds, domain_val, domain_test = list(range(5)), 810730, 820730
    protocol = {"checkpoints": CHECKPOINTS, "first_layer_samples": 64,
                "gamma_candidates": gamma_names, "trained_model_seeds": model_seeds,
                "sampling_seeds_per_model": [0], "validation_examples": 5000,
                "test_examples": 10000, "chain_batch_size": 128,
                "validation_key_domain": domain_val, "test_key_domain": domain_test,
                "selection_rule": "Smallest mean executed free-node update count including chain padding, subject to validation accuracy at least full-horizon accuracy minus 0.002 and CE at most full-horizon CE plus 0.02; ties by CE and candidate order.",
                "test_labels_used_in_selection": False,
                "confidence_scope": "Conditional output-layer empirical SE with first-layer hidden input frozen; output-column weights independent. Heuristic normal margin rule at multiple looks, not a coverage guarantee. First-layer uncertainty is not included.",
                "chain_protocol": "Independent case keys; chunks reset the uncoupled unary bank and perform warmup1. Every chunk record is a fresh stationary draw, costing exactly one sweep. No claim for coupled-bank continuation.",
                "online_execution": "Drop stopped cases and compact unfinished IDs into the next THRML vmap request; pad final chain block to 128 and charge all lanes. No full-horizon call is made for stopped cases in the online branch.",
                "experimental_reuse": "Validation uses full traces for rule search. Test full traces support a reference/replay audit, and are separate from actual online execution. Reusing fixed first-layer hidden values for fixed-budget comparators is charged to every comparator.",
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [source/"evaluation_data.npz"]+[source/f"seed{s}.npz" for s in model_seeds]},
                "intervals": "Approximate 1.96*SD/sqrt(5) across five independently trained model/sampling-seed pairs; one sampling seed per model, no separate within-model Monte Carlo CI",
                "arithmetic": "Float32 control fields, float32 modeled dot products/means/moments/activations, plus deterministic neural biases; no chip latency/energy claim"}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    models, val_rows, val_raw, start = {}, [], {"labels": data["validation_labels"]}, time.monotonic()
    for model in model_seeds:
        z = np.load(source/f"seed{model}.npz")
        w, b = [z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)]
        net = AdaptiveNetwork(w, b)
        models[model] = (net, w, b)
        hidden, first_obs = net.hidden(data["validation_inputs"], domain_val, model, 0)
        history, moments, counters = net.collect(hidden, domain_val, model, 0)
        val_raw[f"model{model}_history"], val_raw[f"model{model}_m2"] = history, moments
        f0, f1 = len(net.first.bank.free_indices), len(net.bank.free_indices)
        for name in gamma_names:
            logits, stop = replay(history, moments, gamma_value(name))
            second_obs = compact_observations(stop, net.batch_size)
            val_rows.append({"model_seed": model, "gamma": name, **metrics(logits, data["validation_labels"]),
                             "mean_output_samples": float(stop.mean()),
                             "first_chain_observations": first_obs, "second_chain_observations": second_obs,
                             "executed_free_node_updates": first_obs*f0+second_obs*f1})
        print(f"adaptive validation model={model} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"validation_metrics.csv", val_rows)
    np.savez_compressed(out/"validation_traces.npz", **val_raw)
    profiles = []
    for name in gamma_names:
        group = [r for r in val_rows if r["gamma"] == name]
        profiles.append({"gamma": name, "accuracy_mean": float(np.mean([r["accuracy"] for r in group])),
                         "cross_entropy_mean": float(np.mean([r["cross_entropy"] for r in group])),
                         "executed_free_updates_mean": float(np.mean([r["executed_free_node_updates"] for r in group])),
                         "mean_output_samples": float(np.mean([r["mean_output_samples"] for r in group]))})
    reference = next(r for r in profiles if r["gamma"] == "never")
    passing = [r for r in profiles if r["accuracy_mean"] >= reference["accuracy_mean"]-.002
               and r["cross_entropy_mean"] <= reference["cross_entropy_mean"]+.02]
    selected = min(passing, key=lambda r: (r["executed_free_updates_mean"], r["cross_entropy_mean"], gamma_names.index(r["gamma"])))
    selection = {"selected": selected, "reference": reference, "profiles": profiles,
                 "test_labels_used_in_selection": False}
    (out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    print("Adaptive rule frozen:", selected, flush=True)
    x, y = data["test_inputs"], data["test_labels"]
    test_rows, raw, stage_counts, audits = [], {"labels": y}, {}, []
    for model, (net, weights, biases) in models.items():
        hidden, first_obs = net.hidden(x, domain_test, model, 0)
        history, moments, reference_counts = net.collect(hidden, domain_test, model, 0)
        # A fresh online first-layer call demonstrates end-to-end execution,
        # instead of treating precomputed traces as actual adaptive savings.
        online_hidden, online_first_obs = net.hidden(x, domain_test, model, 0)
        np.testing.assert_array_equal(online_hidden, hidden)
        online, stop, counts = net.online(online_hidden, domain_test, model, 0, gamma_value(selected["gamma"]))
        expected_online, expected_stop = replay(history, moments, gamma_value(selected["gamma"]))
        np.testing.assert_array_equal(stop, expected_stop)
        np.testing.assert_array_equal(online, expected_online)
        second_obs = sum(c["chain_observations"] for c in counts)
        assert second_obs == compact_observations(stop, net.batch_size)
        padded_n = ((len(x)+net.batch_size-1)//net.batch_size)*net.batch_size
        f0, f1 = len(net.first.bank.free_indices), len(net.bank.free_indices)
        raw[f"model{model}_history"], raw[f"model{model}_m2"] = history, moments
        raw[f"model{model}_hidden"], raw[f"model{model}_online"] = hidden, online
        raw[f"model{model}_stop"] = stop
        raw[f"model{model}_deterministic"] = deterministic(x, weights, biases)
        stage_counts[str(model)] = {"online": counts, "reference": reference_counts,
                                    "first_chain_observations": first_obs, "free_nodes": [f0, f1]}
        outputs = {"full256": (history[-1], sum(c["chain_observations"] for c in reference_counts), 256.),
                   "adaptive": (online, second_obs, float(stop.mean()))}
        for bracket, t in (("matched_floor", second_obs//padded_n),
                           ("matched_ceil", (second_obs+padded_n-1)//padded_n)):
            assert 1 <= t <= 256
            trace, _, fixed_counts = net.collect(hidden, domain_test, model, 0, t)
            outputs[bracket] = (trace[-1], sum(c["chain_observations"] for c in fixed_counts), float(t))
            raw[f"model{model}_{bracket}"] = trace[-1]
        baseline_accuracy = metrics(history[-1], y)["accuracy"]
        for name, (logits, observed, mean_samples) in outputs.items():
            score = metrics(logits, y)
            test_rows.append({"model_seed": model, "condition": name, **score,
                "accuracy_delta_vs_full256": score["accuracy"]-baseline_accuracy,
                "mean_output_samples": mean_samples, "first_chain_observations": first_obs,
                "second_chain_observations": observed,
                "executed_free_node_updates": first_obs*f0+observed*f1,
                "logical_weight_readouts": first_obs*weights[0].size+observed*weights[1].size,
                "actual_second_observations_per_test_case": observed/len(y)})
        audits.append({"model_seed": model, "online_replay_exact": True, "online_first_recomputed_exact": True,
                       "stopped_before256": int(np.sum(stop < 256)), "stop_distribution": {str(t): int(np.sum(stop == t)) for t in CHECKPOINTS},
                       "adaptive_minus_full_mean_logit": (online-history[-1]).mean(axis=0).tolist(),
                       "adaptive_minus_full_logit_rmse": float(np.sqrt(np.mean((online-history[-1])**2))),
                       "prediction_disagreement_with_full256": float(np.mean(online.argmax(1) != history[-1].argmax(1))),
                       "fixed_mean_target_note": "Stopping changes the distribution of returned logits; these differences quantify finite-horizon selection, not an unbiasedness claim."})
        print(f"adaptive test model={model} meanT={stop.mean():.2f} accuracy={metrics(online,y)['accuracy']:.4f} full={baseline_accuracy:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
    write_csv(out/"test_metrics.csv", test_rows)
    np.savez_compressed(out/"test_traces.npz", **raw)
    (out/"stage_counters.json").write_text(json.dumps(stage_counts, indent=2)+"\n")
    (out/"online_audit.json").write_text(json.dumps(audits, indent=2)+"\n")
    summary = []
    for condition in ("full256", "adaptive", "matched_floor", "matched_ceil"):
        group = [r for r in test_rows if r["condition"] == condition]
        a = [r["accuracy"] for r in group]
        d = [r["accuracy_delta_vs_full256"] for r in group]
        ratios = [r["executed_free_node_updates"]/next(s["executed_free_node_updates"] for s in test_rows
                      if s["model_seed"] == r["model_seed"] and s["condition"] == "full256") for r in group]
        paired_adaptive = [next(s["accuracy"] for s in test_rows if s["model_seed"] == r["model_seed"] and s["condition"] == "adaptive")-r["accuracy"] for r in group]
        summary.append({"condition": condition, "accuracy_mean": float(np.mean(a)), "accuracy_ci95_margin": ci(a),
                        "paired_accuracy_delta_vs_full256": float(np.mean(d)), "paired_delta_ci95_margin": ci(d),
                        "adaptive_minus_condition_accuracy_mean": float(np.mean(paired_adaptive)), "adaptive_minus_condition_ci95_margin": ci(paired_adaptive),
                        "mean_output_samples": float(np.mean([r["mean_output_samples"] for r in group])),
                        "mean_executed_free_node_updates": float(np.mean([r["executed_free_node_updates"] for r in group])),
                        "mean_fraction_of_full256_updates": float(np.mean(ratios)),
                        "mean_logical_weight_readouts": float(np.mean([r["logical_weight_readouts"] for r in group]))})
    write_csv(out/"summary.csv", summary)
    manifest(out/"manifest.json", protocol, {"selection": selection, "summary": summary,
            "online_audit": audits, "elapsed_seconds": time.monotonic()-start,
            "controls": {str(m): [net.first.bank.controls(), net.bank.controls()] for m, (net, _, _) in models.items()}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
