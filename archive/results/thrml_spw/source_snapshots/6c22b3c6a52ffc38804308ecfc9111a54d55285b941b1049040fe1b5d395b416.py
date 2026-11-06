"""Raw evidence audit for selective native Transformer projection conversion."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("results/thrml_spw")
    preparation = json.loads((root/"transformer_preparation/manifest.json").read_text())
    export_path = Path("data/thrml_spw/transformer_projection/export.npz")
    assert sha(export_path) == preparation["export_sha256"]
    snapshot = Path(preparation["model_snapshot_path"])
    for name, digest in preparation["model_file_sha256"].items():
        assert sha(snapshot/name) == digest
    export = np.load(export_path)
    native_root = root/"transformer_projection_native"
    native_manifest = json.loads((native_root/"manifest.json").read_text())
    protocol = native_manifest["config"]
    assert protocol["input_export_sha256"] == preparation["export_sha256"]
    assert native_manifest["source_commit"] == "2d54cd8404df85bb65703d7e8b4e84e48c1b72a9"
    assert preparation["converted_matrix_parameters"] == export["weight"].size == 589824
    assert preparation["converted_fraction"] < .01
    weight = export["weight"]
    lower, span = weight.min(0), weight.max(0)-weight.min(0)
    p = np.zeros_like(weight)
    np.divide(weight-lower, span, out=p, where=span != 0)
    p = np.clip(p, 0, 1).astype(np.float64)
    free = (p > 0) & (p < 1)
    fields = ((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32)
    q = p.copy();q[free] = np.exp(-np.logaddexp(0, -2*fields.astype(np.float64)))
    variance = span**2*q*(1-q)
    assert native_manifest["result"]["controls"]["free_pbits"] == int(free.sum())
    native_path = native_root/"outputs.npz"
    outputs = np.load(native_path)
    theory = {}
    for split in ("validation", "test"):
        inputs = export[split+"_inputs"]
        mean_x2 = np.mean(inputs.astype(np.float64)**2, axis=(0, 1))
        theory[split] = float(np.mean(mean_x2@variance))
        assert inputs.shape[1:] == (65, 768)
        offsets = export[split+"_token_offsets"]
        assert np.min(np.diff(offsets)) >= 65
        np.testing.assert_array_equal(outputs[split+"_FP32"], export[split+"_reference_outputs"])
    linear_rows = read(native_root/"linear_metrics.csv")
    for r in linear_rows:
        y = outputs[f"{r['split']}_T{r['samples']}_seed{r['sampling_seed']}"]
        expected = export[r["split"]+"_reference_outputs"]
        mse = float(np.mean((y.astype(np.float64)-expected)**2))
        np.testing.assert_allclose(mse, float(r["linear_mse"]), rtol=1e-12)
        np.testing.assert_allclose(theory[r["split"]]/int(r["samples"]), float(r["predicted_sampling_mse"]), rtol=1e-10)
        assert int(r["free_node_updates"]) == int(r["bank_batches"])*int(r["samples"])*int(free.sum())
    replay = root/"transformer_replay"
    replay_protocol = json.loads((replay/"protocol.json").read_text())
    assert replay_protocol["native_outputs_sha256"] == sha(native_path)
    assert not replay_protocol["test_labels_used_in_selection"]
    raw = np.load(replay/"token_evidence.npz")
    rows = read(replay/"metrics.csv")
    for r in rows:
        key = f"{r['split']}_T{r['samples']}_seed{r['sampling_seed']}" if int(r["samples"]) else f"{r['split']}_{r['condition']}"
        nll, predicted = raw[key+"_token_nll"], raw[key+"_predicted_tokens"]
        ce = float(nll.astype(np.float64).mean())
        np.testing.assert_allclose(ce, float(r["cross_entropy"]), rtol=1e-12)
        np.testing.assert_allclose(np.exp(ce), float(r["perplexity"]), rtol=1e-12)
        np.testing.assert_allclose(np.mean(predicted == raw[r["split"]+"_target_tokens"]), float(r["token_accuracy"]), atol=1e-12)
        if r["condition"] == "FP32":
            np.testing.assert_array_equal(nll, export[r["split"]+"_reference_token_nll"])
    selection = json.loads((replay/"selection.json").read_text())
    baseline = next(r for r in rows if r["split"] == "validation" and r["condition"] == "FP32")
    passing = []
    for p in selection["profiles"]:
        group = [r for r in rows if r["split"] == "validation" and int(r["samples"]) == p["samples"]]
        ce, acc = np.mean([float(r["cross_entropy"]) for r in group]), np.mean([float(r["token_accuracy"]) for r in group])
        np.testing.assert_allclose(ce, p["mean_calibration_ce"], atol=1e-12)
        np.testing.assert_allclose(acc, p["mean_calibration_token_accuracy"], atol=1e-12)
        passed = ce-float(baseline["cross_entropy"]) <= np.log(1.05) and float(baseline["token_accuracy"])-acc <= .01
        assert bool(passed) == p["passes"]
        if passed:passing.append(p)
    assert selection["selected"] == (min(passing, key=lambda r: r["samples"]) if passing else selection["profiles"][-1])
    for r in read(replay/"summary.csv"):
        group = [s for s in rows if s["split"] == r["split"] and s["condition"] == r["condition"]]
        ce = np.array([float(s["cross_entropy"]) for s in group])
        np.testing.assert_allclose(ce.mean(), float(r["cross_entropy_mean"]), atol=1e-12)
        margin = 1.96*ce.std(ddof=1)/np.sqrt(len(ce)) if len(ce) > 1 else 0.
        np.testing.assert_allclose(margin, float(r["cross_entropy_ci95_margin"]), atol=1e-12)
    result = {"passed": True, "native_linear_conditions": len(linear_rows), "downstream_replay_conditions": len(rows),
        "converted_projection_weights": 589824, "converted_fraction": preparation["converted_fraction"],
        "max_expected_weight_rounding_error": float(np.max(np.abs(lower+span*q-weight))),
        "scope": "Pinned model/export bytes, selective affine mapping, native linear scores and counters, exact FP32 token replay, raw language scores and calibration-only gate; not full-model or full-corpus certification."}
    (replay/"evidence_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
