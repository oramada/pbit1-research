"""Audit raw online decisions and shared-bank resource counters independently."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .verify_adaptive import audit_policy, read, score


def counts_from_stop(stop, checkpoints, batch):
    counters, previous = [], 0
    for target in checkpoints:
        active = int(np.sum(stop > previous))
        if active:
            batches = (active+batch-1)//batch
            counters.append({"previous": previous, "target": target, "active_cases": active,
                "batch_count": batches, "padded_cases": batches*batch,
                "bank_observations": batches*(target-previous)})
        previous = target
    return counters


def main():
    out = Path("results/thrml_spw/adaptive_shared")
    protocol = json.loads((out/"protocol.json").read_text())
    selection = json.loads((out/"selection.json").read_text())
    assert not protocol["test_labels_used_in_selection"] and not selection["test_labels_used_in_selection"]
    batch, checkpoints = protocol["batch_size"], protocol["output_checkpoints"]
    val = np.load(out/"validation_outputs.npz")
    val_rows = read(out/"validation_metrics.csv")
    val_counts = json.loads((out/"validation_counters.json").read_text())
    for row in val_rows:
        key = f"model{row['model_seed']}_gamma{row['gamma']}"
        a, ce = score(val[key+"_logits"], val["labels"])
        np.testing.assert_allclose(a, float(row["accuracy"]), atol=1e-12)
        np.testing.assert_allclose(ce, float(row["cross_entropy"]), rtol=1e-6)
        expected_counts = counts_from_stop(val[key+"_stop"], checkpoints, batch)
        assert val_counts[key] == expected_counts
        assert int(row["second_bank_observations"]) == sum(c["bank_observations"] for c in expected_counts)
    for profile in selection["profiles"]:
        group = [r for r in val_rows if r["gamma"] == profile["gamma"]]
        for target, source in (("accuracy_mean", "accuracy"), ("cross_entropy_mean", "cross_entropy"),
                               ("executed_free_updates_mean", "executed_free_node_updates")):
            np.testing.assert_allclose(profile[target], np.mean([float(r[source]) for r in group]))
    ref = selection["reference"]
    passing = [r for r in selection["profiles"] if r["accuracy_mean"] >= ref["accuracy_mean"]-.002 and r["cross_entropy_mean"] <= ref["cross_entropy_mean"]+.02]
    expected = min(passing, key=lambda r: (r["executed_free_updates_mean"], r["cross_entropy_mean"], protocol["gamma_candidates"].index(r["gamma"])))
    assert selection["selected"] == expected
    raw = np.load(out/"test_outputs.npz")
    rows = read(out/"test_metrics.csv")
    counters = json.loads((out/"test_counters.json").read_text())
    for model in range(5):
        weights = np.load(f"data/thrml_spw/positive_mnist/seed{model}.npz")
        free = [int(np.sum((weights[f"weight_{i}"] > 0) & (weights[f"weight_{i}"] < 1))) for i in range(2)]
        for seed in range(5):
            key = f"model{model}_seed{seed}"
            history, moments, stop = raw[key+"_online_history"], raw[key+"_online_m2"], raw[key+"_stop"]
            inferred, inferred_stop = audit_policy(history, moments, selection["selected"]["gamma"], checkpoints)
            np.testing.assert_array_equal(inferred, raw[key+"_adaptive"])
            np.testing.assert_array_equal(inferred_stop, stop)
            for i, target in enumerate(checkpoints):
                active = stop >= target
                assert np.isfinite(history[i, active]).all() and np.isfinite(moments[i, active]).all()
                assert np.isnan(history[i, ~active]).all() and np.isnan(moments[i, ~active]).all()
            expected_counts = counts_from_stop(stop, checkpoints, batch)
            assert counters[key]["online"] == expected_counts
            assert counters[key]["free_nodes"] == free
            batches = (len(stop)+batch-1)//batch
            first_obs = batches*64
            online_obs = sum(c["bank_observations"] for c in expected_counts)
            group = [r for r in rows if int(r["model_seed"]) == model and int(r["sampling_seed"]) == seed]
            for row in group:
                condition = row["condition"]
                a, ce = score(raw[key+"_"+condition], raw["labels"])
                np.testing.assert_allclose(a, float(row["accuracy"]), atol=1e-12)
                np.testing.assert_allclose(ce, float(row["cross_entropy"]), rtol=1e-6)
                second_obs = int(row["second_bank_observations"])
                assert int(row["first_bank_observations"]) == first_obs
                assert int(row["executed_free_node_updates"]) == first_obs*free[0]+second_obs*free[1]
                reads = first_obs*2048+second_obs*320
                assert int(row["logical_weight_readouts"]) == reads
                assert int(row["padded_binary_product_terms"]) == batch*reads
                if condition == "adaptive":
                    assert second_obs == online_obs
                elif condition == "full256":
                    assert second_obs == batches*256
                else:
                    difference = second_obs-online_obs
                    assert abs(difference) < batches
                    assert difference <= 0 if condition == "matched_floor" else difference >= 0
    for summary in read(out/"summary.csv"):
        condition = summary["condition"]
        values, differences, adaptive_differences = [], [], []
        for model in range(5):
            per_model = [r for r in rows if int(r["model_seed"]) == model]
            average = lambda name: np.mean([float(r["accuracy"]) for r in per_model if r["condition"] == name])
            values.append(average(condition))
            differences.append(average(condition)-average("full256"))
            adaptive_differences.append(average("adaptive")-average(condition))
        for field, vector, interval in (("accuracy_mean", values, "accuracy_ci95_margin"),
                ("paired_accuracy_delta_vs_full256", differences, "paired_delta_ci95_margin"),
                ("adaptive_minus_condition_accuracy_mean", adaptive_differences, "adaptive_minus_condition_ci95_margin")):
            np.testing.assert_allclose(float(summary[field]), np.mean(vector), atol=1e-12)
            np.testing.assert_allclose(float(summary[interval]), 1.96*np.std(vector, ddof=1)/np.sqrt(5), atol=1e-12)
    result = {"passed": True, "validation_conditions": len(val_rows), "test_conditions": len(rows),
              "trained_models": 5, "sampling_seeds_per_model": 5,
              "scope": "Raw scores, frozen selection, actual online decision traces, stopped-case compaction, bank/batch/padding counts, matched updates and five-model interval aggregation. No physical or confidence-coverage certification."}
    (out/"evidence_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
