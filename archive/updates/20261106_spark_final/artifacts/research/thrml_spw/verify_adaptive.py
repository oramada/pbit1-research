"""Independent saved-trace audit of policy decisions, costs, and raw scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def score(logits, labels):
    z = logits.astype(np.float64)
    maximum = z.max(axis=1)
    normalizer = maximum+np.log(np.exp(z-maximum[:, None]).sum(axis=1))
    return float(np.mean(z.argmax(1) == labels)), float(np.mean(normalizer-z[np.arange(len(z)), labels]))


def audit_policy(history, moments, gamma, checkpoints):
    # Reimplement the recorded rule from raw moments, without the runtime helper.
    values = np.zeros_like(history[0])
    stop = np.zeros(len(values), dtype=np.int32)
    for index, n in enumerate(checkpoints):
        logits = history[index]
        columns = np.argsort(logits, axis=1)[:, -2:]
        i = np.arange(len(values))
        a, b = columns[:, 1], columns[:, 0]
        margin = logits[i, a]-logits[i, b]
        se = np.sqrt(np.maximum(moments[index, i, a]+moments[index, i, b], 0)/(n*(n-1)))
        accepted = (margin > float(gamma)*se) if gamma != "never" else np.zeros(len(values), dtype=bool)
        accepted &= stop == 0
        if n == checkpoints[-1]:
            accepted = stop == 0
        values[accepted], stop[accepted] = logits[accepted], n
    return values, stop


def padded_observations(stop, batch, checkpoints):
    previous, total = 0, 0
    stages = []
    for target in checkpoints:
        remaining = int(np.sum(stop > previous))
        padded = (remaining+batch-1)//batch*batch
        total += padded*(target-previous)
        if remaining:
            stages.append({"previous": previous, "target": target, "active_cases": remaining,
                           "padded_cases": padded, "chain_observations": padded*(target-previous)})
        previous = target
    return total, stages


def main():
    out = Path("results/thrml_spw/adaptive_native")
    protocol = json.loads((out/"protocol.json").read_text())
    selected = json.loads((out/"selection.json").read_text())
    assert not protocol["test_labels_used_in_selection"] and not selected["test_labels_used_in_selection"]
    checkpoints, batch = protocol["checkpoints"], protocol["chain_batch_size"]
    val = np.load(out/"validation_traces.npz")
    val_rows = read(out/"validation_metrics.csv")
    for row in val_rows:
        model, gamma = row["model_seed"], row["gamma"]
        logits, stop = audit_policy(val[f"model{model}_history"], val[f"model{model}_m2"], gamma, checkpoints)
        a, ce = score(logits, val["labels"])
        np.testing.assert_allclose(a, float(row["accuracy"]), atol=1e-12)
        np.testing.assert_allclose(ce, float(row["cross_entropy"]), rtol=1e-6, atol=1e-8)
        observations, _ = padded_observations(stop, batch, checkpoints)
        assert observations == int(row["second_chain_observations"])
    for profile in selected["profiles"]:
        group = [r for r in val_rows if r["gamma"] == profile["gamma"]]
        for dest, source in (("accuracy_mean", "accuracy"), ("cross_entropy_mean", "cross_entropy"),
                             ("executed_free_updates_mean", "executed_free_node_updates")):
            np.testing.assert_allclose(profile[dest], np.mean([float(r[source]) for r in group]))
    reference = selected["reference"]
    passing = [r for r in selected["profiles"] if r["accuracy_mean"] >= reference["accuracy_mean"]-.002
               and r["cross_entropy_mean"] <= reference["cross_entropy_mean"]+.02]
    expected = min(passing, key=lambda r: (r["executed_free_updates_mean"], r["cross_entropy_mean"], protocol["gamma_candidates"].index(r["gamma"])))
    assert expected == selected["selected"]
    test = np.load(out/"test_traces.npz")
    counters = json.loads((out/"stage_counters.json").read_text())
    test_rows = read(out/"test_metrics.csv")
    for model in protocol["trained_model_seeds"]:
        weights = np.load(f"data/thrml_spw/positive_mnist/seed{model}.npz")
        free = [int(np.sum((weights[f"weight_{i}"] > 0) & (weights[f"weight_{i}"] < 1))) for i in range(2)]
        assert free == counters[str(model)]["free_nodes"]
        logits, stop = audit_policy(test[f"model{model}_history"], test[f"model{model}_m2"], selected["selected"]["gamma"], checkpoints)
        np.testing.assert_array_equal(logits, test[f"model{model}_online"])
        np.testing.assert_array_equal(stop, test[f"model{model}_stop"])
        second_observations, expected_stages = padded_observations(stop, batch, checkpoints)
        assert expected_stages == counters[str(model)]["online"]
        padded_n = ((len(stop)+batch-1)//batch)*batch
        first_observations = padded_n*64
        for row in (r for r in test_rows if int(r["model_seed"]) == model):
            condition = row["condition"]
            values = test[f"model{model}_online"] if condition == "adaptive" else (
                test[f"model{model}_history"][-1] if condition == "full256" else test[f"model{model}_{condition}"])
            a, ce = score(values, test["labels"])
            np.testing.assert_allclose(a, float(row["accuracy"]), atol=1e-12)
            np.testing.assert_allclose(ce, float(row["cross_entropy"]), rtol=1e-6, atol=1e-8)
            observed = int(row["second_chain_observations"])
            assert int(row["first_chain_observations"]) == first_observations
            assert int(row["executed_free_node_updates"]) == first_observations*free[0]+observed*free[1]
            assert int(row["logical_weight_readouts"]) == first_observations*2048+observed*320
            if condition == "adaptive":
                assert observed == second_observations
            elif condition == "full256":
                assert observed == padded_n*256
            else:
                difference = observed-second_observations
                assert abs(difference) < padded_n
                assert difference <= 0 if condition == "matched_floor" else difference >= 0
    bias_path = Path("results/thrml_spw/adaptive_bias")
    bias_raw = np.load(bias_path/"prefix_counts.npz")
    for row in read(bias_path/"metrics.csv"):
        initial, final = bias_raw["initial16"].astype(float), bias_raw["final256"].astype(float)
        m = initial/16
        stop = np.abs(m-.5) > float(row["gamma"])*np.sqrt(m*(1-m)/15) if row["gamma"] != "never" else np.zeros(len(m), dtype=bool)
        values = np.where(stop, m, final/256)
        np.testing.assert_allclose(values.mean(), float(row["empirical_mean"]), atol=1e-12)
        assert abs(values.mean()-float(row["exact_returned_mean"])) < 6*float(row["empirical_mean_se"])
    result = {"passed": True, "validation_conditions": len(val_rows), "test_conditions": len(test_rows),
              "online_policy_and_compaction_models": 5, "exact_bias_control_trials": 10000,
              "scope": "Raw scores, frozen validation selection, case stopping, padding-inclusive sampler work, matched budgets, native bias-law check. No hardware or confidence-coverage certification."}
    (out/"evidence_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
