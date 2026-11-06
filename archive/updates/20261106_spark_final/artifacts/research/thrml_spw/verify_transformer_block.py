"""Independent raw array, lineage, cost, gate and uncertainty block audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):h.update(block)
    return h.hexdigest()


def scores(nll, predicted, tokens):
    ce = float(nll.astype(np.float64).mean())
    return ce, float(np.mean(predicted == tokens[:, 1:]))


def main():
    parser = argparse.ArgumentParser();parser.add_argument("--root", default="results/thrml_spw/transformer_block0_gpu")
    out = Path(parser.parse_args().root)
    protocol = json.loads((out/"protocol.json").read_text())
    final = json.loads((out/"manifest.json").read_text())
    selection = json.loads((out/"selection.json").read_text())
    assert not protocol["test_labels_used_in_selection"] and not selection["test_labels_used_in_selection"]
    data_path = Path("data/thrml_spw/transformer_projection/export.npz")
    assert sha(data_path) == protocol["input_export_sha256"]
    data = np.load(data_path);weights = np.load(out/"weights.npz")
    assert sum(weights[f"weight_{i}"].size for i in range(4)) == protocol["converted_matrix_weights"] == 7077888
    weight_sha = sha(out/"weights.npz");assert weight_sha == final["weights_sha256"]
    properties = []
    for i in range(4):
        w = weights[f"weight_{i}"]
        if protocol.get("grouping", "").startswith("per-output input blocks of 128"):
            lower = np.repeat(np.stack([w[j:j+128].min(0) for j in range(0,len(w),128)]),128,axis=0)[:len(w)]
            upper = np.repeat(np.stack([w[j:j+128].max(0) for j in range(0,len(w),128)]),128,axis=0)[:len(w)]
            span = upper-lower
            groups = ((len(w)+127)//128)*w.shape[1]
        else:
            lower, span = w.min(0), w.max(0)-w.min(0)
            groups = w.shape[1]
        p = np.zeros_like(w);np.divide(w-lower, span, out=p, where=span != 0);p = np.clip(p, 0, 1).astype(np.float64)
        free = (p > 0) & (p < 1)
        b = ((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32)
        q = p.copy();q[free] = np.exp(-np.logaddexp(0, -2*b.astype(np.float64)))
        properties.append({"free": int(free.sum()), "weights": w.size,
                           "variance": span**2*q*(1-q), "groups": groups})
    baseline_raw = np.load(out/"baseline_tokens.npz")
    baseline_rows = read(out/"baseline_metrics.csv")
    for row in baseline_rows:
        split, name = row["split"], row["condition"]
        ce, acc = scores(baseline_raw[f"{split}_{name}_nll"], baseline_raw[f"{split}_{name}_predictions"], data[split+"_tokens"])
        np.testing.assert_allclose(ce, float(row["cross_entropy"]), atol=1e-12)
        np.testing.assert_allclose(acc, float(row["token_accuracy"]), atol=1e-12)
        if name == "FP32":np.testing.assert_array_equal(baseline_raw[f"{split}_{name}_nll"], data[split+"_reference_token_nll"])
    rows, prefixes = read(out/"native_metrics.csv"), read(out/"prefix_metrics.csv")
    expected_calibration = {(t, seed) for profile in selection["profiles"] for t in [profile["samples"]] for seed in range(5)}
    expected_test = {(t, seed) for t in {64, selection["selected"]["samples"]} for seed in range(5)}
    assert {(int(r["samples"]),int(r["sampling_seed"])) for r in rows if r["split"] == "validation"} == expected_calibration
    assert {(int(r["samples"]),int(r["sampling_seed"])) for r in rows if r["split"] == "test"} == expected_test
    stage_count = 0;max_reference_scaled_error = 0.;max_mse_discrepancy = 0.
    for row in rows:
        split, t, seed = row["split"], int(row["samples"]), int(row["sampling_seed"])
        directory = out/"runs"/f"{split}_T{t}_seed{seed}"
        record = json.loads((directory/"run.json").read_text());assert record["status"] == "complete"
        raw = np.load(directory/"prefix_tokens.npz")
        ce, acc = scores(raw["prefix4_nll"], raw["prefix4_predictions"], data[split+"_tokens"])
        np.testing.assert_allclose(ce, float(row["cross_entropy"]), atol=1e-12)
        np.testing.assert_allclose(acc, float(row["token_accuracy"]), atol=1e-12)
        assert record["exact_prefix_input_checks"] == 10*len(data[split+"_tokens"])
        total_updates = 0
        for i, stage in enumerate(record["stages"]):
            meta = json.loads(Path(stage["metadata"]).read_text());assert meta["status"] == "complete"
            signature = meta["signature"]
            assert signature["weights_sha256"] == weight_sha
            assert signature["samples"] == t and signature["stage"] == i and signature["seed"] == seed
            assert signature.get("jax_backend") == "gpu" and signature.get("matmul_precision") == "highest"
            for path, key in ((stage["inputs"],"inputs_sha256"),(stage["reference"],"reference_sha256")):
                assert sha(path) == signature[key]
            assert sha(stage["output"]) == meta["output_sha256"]
            x, reference, y = [np.load(stage[key], mmap_mode="r") for key in ("inputs", "reference", "output")]
            expected = x@weights[f"weight_{i}"]+weights[f"bias_{i}"]
            error = float(np.max(np.abs(expected-reference)))/(1+float(np.max(np.abs(reference))))
            max_reference_scaled_error = max(max_reference_scaled_error,error);assert error < 2e-5
            mse = float(np.mean((y.astype(np.float64)-reference)**2))
            np.testing.assert_allclose(mse, meta["conditional_linear_mse"], rtol=1e-12)
            prediction = float(np.mean(np.mean(x.astype(np.float64)**2, axis=(0,1))@properties[i]["variance"])/t)
            np.testing.assert_allclose(prediction,meta["predicted_sampling_mse"],rtol=1e-10)
            max_mse_discrepancy = max(max_mse_discrepancy,abs(mse/prediction-1))
            batches = (len(x)+7)//8
            assert meta["free_node_updates"] == batches*t*properties[i]["free"]
            assert meta["logical_weight_readouts"] == batches*t*properties[i]["weights"]
            assert meta["controls"]["metadata_bits"] == 2*properties[i]["groups"]*32
            total_updates += meta["free_node_updates"]
            assert stage["prior_native_output_sha256"] == [sha(directory/f"stage{j}_native_output.npy") for j in range(i)]
            teacher = np.load(out/"teacher"/f"{split}_stage{i}_inputs.npy", mmap_mode="r")
            rmse = float(np.sqrt(np.mean((x.astype(np.float64)-teacher)**2)))
            np.testing.assert_allclose(rmse,stage["student_input_rmse_vs_teacher"],rtol=1e-12,atol=1e-12)
            if i == 0:assert rmse == 0
            else:assert rmse > 0
            stage_count += 1
        assert total_updates == int(row["free_node_updates"]) == record["free_node_updates"]
        for p in range(5):
            reported = next(r for r in prefixes if r["split"] == split and int(r["samples"]) == t and int(r["sampling_seed"]) == seed and int(r["converted_prefix"]) == p)
            nll_ce, nll_acc = scores(raw[f"prefix{p}_nll"],raw[f"prefix{p}_predictions"],data[split+"_tokens"])
            np.testing.assert_allclose(nll_ce,float(reported["cross_entropy"]),atol=1e-12)
            np.testing.assert_allclose(nll_acc,float(reported["token_accuracy"]),atol=1e-12)
    reference = next(r for r in baseline_rows if r["split"] == "validation" and r["condition"] == "FP32")
    for profile in selection["profiles"]:
        group = [r for r in rows if r["split"] == "validation" and int(r["samples"]) == profile["samples"]]
        ce, acc = np.mean([float(r["cross_entropy"]) for r in group]),np.mean([float(r["token_accuracy"]) for r in group])
        passed = bool(ce-float(reference["cross_entropy"]) <= np.log(1.05) and float(reference["token_accuracy"])-acc <= .01)
        assert passed == profile["passes"]
        np.testing.assert_allclose(ce,profile["cross_entropy_mean"],atol=1e-12)
    passing = [r for r in selection["profiles"] if r["passes"]]
    assert selection["selected"] == (passing[0] if passing else selection["profiles"][-1])
    for summary in read(out/"summary.csv"):
        group = [r for r in rows if r["split"] == summary["split"] and int(r["samples"]) == int(summary["samples"])]
        values = np.array([float(r["cross_entropy"]) for r in group])
        np.testing.assert_allclose(values.mean(),float(summary["cross_entropy_mean"]),atol=1e-12)
        np.testing.assert_allclose(1.96*values.std(ddof=1)/np.sqrt(5),float(summary["cross_entropy_ci95_margin"]),atol=1e-12)
        assert int(summary["sequential_sampling_rounds"]) == 4*int(summary["samples"])
    hook_audit = json.loads((out/"independent_hook_audit.json").read_text())
    assert hook_audit["passed"] and hook_audit["completed_runs_audited"] == len(rows)
    result = {"passed": True,"complete_native_runs":len(rows),"native_operator_checkpoints":stage_count,
        "exact_independent_hook_runs":hook_audit["completed_runs_audited"],"max_conditional_reference_scaled_error":max_reference_scaled_error,
        "max_single_run_mse_fraction_difference_from_analytic_prediction":max_mse_discrepancy,
        "converted_parameters":protocol["converted_matrix_weights"],"converted_fraction":protocol["converted_fraction"],
        "scope":"Complete declared calibration/test schedules, raw student arrays and causal lineage, conditional linear references, native MSE/counters, token losses/gate/uncertainty and independent hooks. Not whole-model or physical certification."}
    (out/"evidence_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":main()
