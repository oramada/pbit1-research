"""Recompute allocation/accumulator evidence from saved raw outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

def metrics(logits, labels):
    """Independent float64 formula, rather than the experiment's SciPy helper."""
    logits = np.asarray(logits, dtype=np.float64)
    maximum = logits.max(axis=1)
    normalizer = maximum+np.log(np.exp(logits-maximum[:, None]).sum(axis=1))
    return {"accuracy": float(np.mean(logits.argmax(1) == labels)),
            "cross_entropy": float(np.mean(normalizer-logits[np.arange(len(labels)), labels]))}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def check_rows(root, filename, archive, key_fn):
    raw = np.load(root/archive)
    rows = read(root/filename)
    for row in rows:
        scores = metrics(raw[key_fn(row)], raw["labels"])
        for field in ("accuracy", "cross_entropy"):
            np.testing.assert_allclose(scores[field], float(row[field]), rtol=1e-6, atol=1e-8)
    return rows


def main():
    base = Path("results/thrml_spw")
    budget = base/"layer_budgets"
    selection = json.loads((budget/"selection.json").read_text())
    assert selection["test_labels_used_in_selection"] is False
    val = check_rows(budget, "validation_metrics.csv", "validation_logits.npz",
                     lambda r: f"model{r['model_seed']}_seed{r['sampling_seed']}_T{r['T0']}_{r['T1']}")
    for profile in selection["profiles"]:
        group = [r for r in val if int(r["T0"]) == profile["T0"] and int(r["T1"]) == profile["T1"]]
        assert len(group) == 10
        for field in ("accuracy", "cross_entropy"):
            np.testing.assert_allclose(np.mean([float(r[field]) for r in group]), profile[field+"_mean"])
    reference = selection["reference"]
    passing = [r for r in selection["profiles"] if r["accuracy_mean"] >= reference["accuracy_mean"]-.005
               and r["cross_entropy_mean"] <= reference["cross_entropy_mean"]+.02]
    chosen = min(passing, key=lambda r: (r["logical_reads_per_example"], r["cross_entropy_mean"], r["T0"], r["T1"]))
    assert chosen == selection["selected"]
    test = check_rows(budget, "test_metrics.csv", "test_logits.npz",
                      lambda r: f"model{r['model_seed']}_seed{r['sampling_seed']}_T{r['T0']}_{r['T1']}")
    for r in test:
        assert int(r["logical_reads_per_isolated_example"]) == 2048*int(r["T0"])+320*int(r["T1"])
        assert int(r["actual_dataset_free_node_updates"]) == 313*int(r["free_node_updates_per_isolated_example"])
    matched = base/"layer_budget_matched"
    rule = json.loads((matched/"protocol.json").read_text())
    assert rule["parent_selection_sha256"] == hashlib.sha256((budget/"selection.json").read_bytes()).hexdigest()
    matched_rows = check_rows(matched, "test_metrics.csv", "test_logits.npz",
                              lambda r: f"model{r['model_seed']}_seed{r['sampling_seed']}_T{r['uniform_samples']}")
    parent_test = np.load(budget/"test_logits.npz")
    for r in matched_rows:
        m, s = r["model_seed"], r["sampling_seed"]
        selected_accuracy = metrics(parent_test[f"model{m}_seed{s}_T{chosen['T0']}_{chosen['T1']}"], parent_test["labels"])["accuracy"]
        np.testing.assert_allclose(selected_accuracy-float(r["accuracy"]), float(r["selected_accuracy_minus_uniform"]))
        assert abs(int(r["uniform_read_difference_from_selected"])) < 2368
    updates = base/"layer_update_matched"
    update_rule = json.loads((updates/"protocol.json").read_text())
    assert update_rule["selection_uses_test_labels"] is False
    update_rows = check_rows(updates, "test_metrics.csv", "test_logits.npz",
                             lambda r: f"model{r['model_seed']}_seed{r['sampling_seed']}_T{r['uniform_samples']}")
    for r in update_rows:
        config = update_rule["budgets_by_model"][r["model_seed"]]
        difference = int(r["update_difference_from_selected"])
        assert abs(difference) < sum(config["free_nodes"])
        assert difference <= 0 if r["bracket"] == "floor" else difference >= 0
        assert int(r["free_updates_per_isolated_example"]) == sum(config["free_nodes"])*int(r["uniform_samples"])
        selected_accuracy = metrics(parent_test[f"model{r['model_seed']}_seed{r['sampling_seed']}_T{chosen['T0']}_{chosen['T1']}"], parent_test["labels"])["accuracy"]
        np.testing.assert_allclose(selected_accuracy-float(r["accuracy"]), float(r["selected_minus_uniform_accuracy"]))
    acc = base/"accumulator_precision"
    acc_sel = json.loads((acc/"selection.json").read_text())
    assert acc_sel["test_labels_used_in_selection"] is False
    acc_val = check_rows(acc, "validation_metrics.csv", "validation_logits.npz",
                         lambda r: f"model{r['model_seed']}_{r['configuration']}")
    for name, profile in acc_sel["profiles"].items():
        group = [r for r in acc_val if r["configuration"] == name]
        assert len(group) == 5
        for field in ("accuracy", "cross_entropy"):
            np.testing.assert_allclose(np.mean([float(r[field]) for r in group]), profile[field])
    base_profile = acc_sel["profiles"]["float32"]
    best = [min((r for r in acc_sel["profiles"].values() if r["bits"] == bits),
                key=lambda r: (r["cross_entropy"], r["fractional_bits"])) for bits in (16, 24)]
    passing = [r for r in best if base_profile["accuracy"]-r["accuracy"] <= .002 and r["cross_entropy"]-base_profile["cross_entropy"] <= .01]
    assert acc_sel["selected"] == (min(passing, key=lambda r: r["bits"]) if passing else base_profile)
    acc_test = check_rows(acc, "test_metrics.csv", "test_logits.npz",
                          lambda r: f"model{r['model_seed']}_seed{r['sampling_seed']}_{r['configuration']}")
    joint = base/"joint_efficiency"
    joint_rule = json.loads((joint/"protocol.json").read_text())
    assert joint_rule["selection_uses_test_labels"] is False
    for name, path in {"allocation": budget/"selection.json", "accumulator": acc/"selection.json",
                       "control": base/"control_neural/selection.json"}.items():
        assert joint_rule["selection_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
    joint_test = check_rows(joint, "test_metrics.csv", "test_logits.npz",
                            lambda r: f"model{r['model_seed']}_seed{r['sampling_seed']}_{r['configuration']}")
    joint_raw = np.load(joint/"test_logits.npz")
    for r in joint_test:
        m, s = r["model_seed"], r["sampling_seed"]
        reference_logits = parent_test[f"model{m}_seed{s}_T256_256"]
        reference_accuracy = metrics(reference_logits, parent_test["labels"])["accuracy"]
        np.testing.assert_allclose(float(r["accuracy"])-reference_accuracy, float(r["accuracy_delta_vs_uniform256"]))
        if r["configuration"] == "field_float32_sum_float32":
            np.testing.assert_array_equal(joint_raw[f"model{m}_seed{s}_{r['configuration']}"],
                parent_test[f"model{m}_seed{s}_T{chosen['T0']}_{chosen['T1']}"])
    bias = np.load(base/"accumulator_bias/sample_means.npz")
    for r in read(base/"accumulator_bias/metrics.csv"):
        values = bias[f"frac{r['fractional_bits']}_T{r['samples']}"]
        assert len(values) == 1000
        mse = np.mean((values.astype(np.float64)-float(r["deterministic_target"]))**2)
        np.testing.assert_allclose(mse, float(r["empirical_mse"]))
        assert abs(mse-float(r["exact_mse"]))/float(r["empirical_mse_se"]) < 6
    result = {"passed": True, "allocation_validation_conditions": len(val),
              "allocation_test_conditions": len(test), "matched_uniform_test_conditions": len(matched_rows),
              "update_matched_test_conditions": len(update_rows),
              "accumulator_validation_conditions": len(acc_val), "accumulator_test_conditions": len(acc_test),
              "joint_confirmation_test_conditions": len(joint_test), "rounding_bias_conditions": 9,
              "scope": "Raw accuracies/CE, validation choices, read/update arithmetic and paired exact-budget comparisons; no physical performance or full-goal certification"}
    (base/"efficiency_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
