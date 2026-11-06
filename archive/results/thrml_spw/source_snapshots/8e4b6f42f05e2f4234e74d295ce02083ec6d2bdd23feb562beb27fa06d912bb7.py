"""Independent score, selection, export and accounting audit for Fashion evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def forward(x, w, b):
    return np.maximum(x@w[0]+b[0], 0)@w[1]+b[1]


def scores(logits, labels):
    values = logits.astype(np.float64)
    maximum = values.max(1)
    normalizer = maximum+np.log(np.exp(values-maximum[:, None]).sum(1))
    return float(np.mean(values.argmax(1) == labels)), float(np.mean(normalizer-values[np.arange(len(values)), labels]))


def main():
    root, source = Path("results/thrml_spw"), Path("data/thrml_spw/positive_fashion")
    training = json.loads((root/"fashion_training/manifest.json").read_text())
    histories = json.loads((root/"fashion_training/history.json").read_text())
    data = np.load(source/"evaluation_data.npz")
    assert len(data["test_labels"]) == 10000 and len(data["validation_labels"]) == 5000
    assert len(np.intersect1d(data["train_indices"], data["validation_indices"])) == 0
    assert len(np.unique(np.concatenate([data["train_indices"], data["validation_indices"]]))) == 60000
    for model in range(5):
        pair = [r for r in training["models"] if r["seed"] == model]
        assert len({r["initial_state_sha256"] for r in pair}) == 1
    for record in training["models"]:
        group = [r for r in histories if r["seed"] == record["seed"] and r["condition"] == record["condition"]]
        best = min(group, key=lambda r: r["validation_ce"])
        assert best["epoch"] == record["selected_epoch"]
        z = np.load(source/f"seed{record['seed']}_{record['condition']}.npz")
        w, b = [z[f"weight_{i}"] for i in range(2)], [z[f"bias_{i}"] for i in range(2)]
        if record["condition"] == "positive_projected":
            assert all(np.all((wi >= 0) & (wi <= 1)) for wi in w)
        else:
            assert any(np.any(wi < 0) for wi in w)
        expected = forward(data["test_inputs"], w, b)
        np.testing.assert_allclose(expected, z["reference_test_logits"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(scores(expected, data["test_labels"])[0], record["test_accuracy"], atol=1e-12)
    out = root/"fashion_native"
    protocol = json.loads((out/"protocol.json").read_text())
    selection = json.loads((out/"selection.json").read_text())
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["source_commit"] == "2d54cd8404df85bb65703d7e8b4e84e48c1b72a9"
    assert not protocol["test_labels_used_in_selection"] and not selection["test_labels_used_in_selection"]
    for path, digest in protocol["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    val = np.load(out/"validation_logits.npz")
    val_rows = read(out/"validation_metrics.csv")
    for row in val_rows:
        logits = val[f"model{row['model_seed']}_{row['route']}_T{row['T0']}_{row['T1']}"]
        a, ce = scores(logits, val["labels"])
        np.testing.assert_allclose(a, float(row["accuracy"]), atol=1e-12)
        np.testing.assert_allclose(ce, float(row["cross_entropy"]), rtol=1e-6, atol=1e-8)
    for route, route_selection in selection["routes"].items():
        for profile in route_selection["profiles"]:
            group = [r for r in val_rows if r["route"] == route and int(r["T0"]) == profile["T0"] and int(r["T1"]) == profile["T1"]]
            assert len(group) == 5
            for dest, src in (("accuracy_mean", "accuracy"), ("cross_entropy_mean", "cross_entropy"), ("free_updates_mean", "free_updates_per_isolated_example")):
                np.testing.assert_allclose(profile[dest], np.mean([float(r[src]) for r in group]))
        passing = [r for r in route_selection["profiles"] if r["accuracy_mean"] >= r["FP32_accuracy_mean"]-.01 and r["cross_entropy_mean"] <= r["FP32_cross_entropy_mean"]+.05]
        expected = min(passing, key=lambda r: (r["free_updates_mean"], r["cross_entropy_mean"], r["T0"], r["T1"])) if passing else min(route_selection["profiles"], key=lambda r: r["cross_entropy_mean"])
        assert bool(passing) == route_selection["gate_passed_on_validation"]
        assert expected == route_selection["selected"]
    raw = np.load(out/"test_logits.npz")
    rows = read(out/"test_metrics.csv")
    for row in rows:
        key = f"model{row['model_seed']}_{row['route']}_{row['condition']}_seed{row['sampling_seed']}"
        a, ce = scores(raw[key], raw["labels"])
        np.testing.assert_allclose(a, float(row["accuracy"]), atol=1e-12)
        np.testing.assert_allclose(ce, float(row["cross_entropy"]), rtol=1e-6, atol=1e-8)
        controls = manifest["result"]["controls"][f"model{row['model_seed']}_{row['route']}"]
        free = [c["free_pbits"] for c in controls]
        t0, t1 = int(row["T0"]), int(row["T1"])
        assert int(row["free_updates_per_isolated_example"]) == free[0]*t0+free[1]*t1
        assert int(row["dataset_free_node_updates"]) == 79*(free[0]*t0+free[1]*t1)
        assert int(row["logical_readouts_per_isolated_example"]) == 12544*t0+640*t1
        if row["condition"] == "matched_updates_ceil_native":
            chosen = selection["routes"][row["route"]]["selected"]
            target = free[0]*chosen["T0"]+free[1]*chosen["T1"]
            assert 0 <= free[0]*t0+free[1]*t1-target < sum(free)
    for summary in read(out/"summary.csv"):
        route, condition = summary["route"], summary["condition"]
        a, d, paired = [], [], []
        for model in range(5):
            group = [r for r in rows if r["route"] == route and r["condition"] == condition and int(r["model_seed"]) == model]
            a.append(np.mean([float(r["accuracy"]) for r in group]))
            d.append(np.mean([float(r["accuracy_delta_vs_FP32"]) for r in group]))
            reference = [r for r in rows if r["route"] == route and r["condition"] == "matched_updates_ceil_native" and int(r["model_seed"]) == model]
            paired.append(a[-1]-np.mean([float(r["accuracy"]) for r in reference]))
        for field, interval, values in (("accuracy_mean", "accuracy_ci95_margin", a),
                ("paired_accuracy_delta_vs_FP32", "paired_delta_ci95_margin", d),
                ("paired_accuracy_delta_vs_matched_ceil", "paired_matched_ci95_margin", paired)):
            np.testing.assert_allclose(float(summary[field]), np.mean(values), atol=1e-12)
            np.testing.assert_allclose(float(summary[interval]), 1.96*np.std(values, ddof=1)/np.sqrt(5), atol=1e-12)
    ledger = read(out/"storage_ledger.csv")
    for row in ledger:
        condition, route = row["condition"], row["route"]
        native = condition.endswith("_native")
        if native:
            controls = manifest["result"]["controls"][f"model{row['model_seed']}_{route}"]
            codes = sum(c["free_pbits"] for c in controls)*32
            metadata = 0 if route == "positive_direct" else 2*74*32
            biases, modes = 74*32, 13184*2
        elif condition in ("FP32", "FP16_rounded_values"):
            bits = 32 if condition == "FP32" else 16
            codes, metadata, biases, modes = 13184*bits, 0, 74*bits, 0
        else:
            bits = int(condition.split("_")[0].replace("affine", ""))
            groups = 74 if condition.endswith("per_output") else 2
            codes, metadata, biases, modes = 13184*bits, groups*2*32, 74*32, 0
        assert int(row["control_field_or_numeric_code_bits"]) == codes
        assert int(row["range_or_quantizer_metadata_bits"]) == metadata
        assert int(row["neural_bias_bits"]) == biases
        assert int(row["endpoint_mode_bits_assumed"]) == modes
        assert int(row["declared_digital_subtotal_excludes_pbit_states"]) == codes+metadata+biases+modes
    result = {"passed": True, "training_checkpoints": 10, "validation_conditions": len(val_rows),
              "test_conditions": len(rows), "storage_ledger_rows": len(ledger), "trained_models_per_condition": 5, "test_sampling_seeds": 5,
              "scope": "Held-out split, checkpoint selection, paired initialization, exported [0,1]/signed weights, native raw scores, validation budget gates, update/read counts and five-model uncertainty. Not hardware certification."}
    (out/"evidence_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
