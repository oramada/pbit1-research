"""Independent checks of raw numerical evidence and reporting invariants.

Passing this bounded audit is not a claim that the entire long-horizon goal or
any physical performance criterion is complete.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .coupling_calibration import exact_moments


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    root=Path("results/thrml_spw");checks={}
    scalar=np.load(root/"scalar_full/prefix_counts.npz")
    counts=scalar["counts"];budgets=scalar["budgets"];weights=scalar["weights"]
    assert counts.shape==(5,200,12,1001)
    assert np.all(counts<=budgets[None,None,:,None])
    assert np.all(counts[:,:,:,0]==0) and np.all(counts[:,:,:,-1]==budgets[None,None,:])
    flat=counts.reshape(-1,len(budgets),len(weights)).astype(np.float64)
    scalar_rows=read(root/"scalar_full/sampling_metrics.csv")
    for ti,t in enumerate(budgets):
        mse=float(np.mean((flat[:,ti]/t-weights)**2))
        reported=next(r for r in scalar_rows if int(r["samples"])==int(t))
        np.testing.assert_allclose(mse,float(reported["rmse"])**2,rtol=1e-12)
    i=int(np.argmin(np.abs(weights-.73)))
    checks["scalar_raw_recomputed"]={"trials_per_target":1000,"target_count":1001,
        "observed_p073":float(np.mean(flat[:,-1,i]/budgets[-1]))}
    selected=json.loads((root/"control_neural/selection.json").read_text())
    assert selected["test_labels_used_in_selection"] is False
    passing=[r for r in selected["choices"] if r["accuracy_drop"]<=.005 and r["cross_entropy_increase"]<=.02]
    assert min(passing,key=lambda r:r["field_bits"])==selected["selected"]
    neural=np.load(root/"control_neural/predictions.npz")
    for r in read(root/"control_neural/test_metrics.csv"):
        predictions=neural[f"seed{r['seed']}_T{r['samples_per_layer']}"]
        np.testing.assert_allclose(np.mean(predictions.argmax(1)==neural["labels"]),float(r["accuracy"]),atol=1e-12)
    checks["selected_control_and_predictions"]={"bits":selected["selected"]["field_bits"],"bound":selected["selected"]["field_bound"],"test_examples":len(neural["labels"])}
    matched=read(root/"matched_resources/matched_statistics.csv")
    for k in (2,4,8):
        for t in (1,4,16):
            pair=[r for r in matched if int(r["K_replicas"])==k and int(r["T_parallel_samples"])==t]
            assert len(pair)==2
            assert int(pair[0]["logical_node_update_events_total"])==int(pair[1]["logical_node_update_events_total"])
            assert int(pair[0]["read_values_total_including_clamps"])==int(pair[1]["read_values_total_including_clamps"])
            assert sorted(int(r["parallel_steps_total"]) for r in pair)==[t,k*t]
            raw=np.load(root/f"matched_resources/raw_K{k}_T{t}.npz")
            for r in pair:
                field="temporal_success_counts" if r["mode"]=="one_temporal_pbit" else "spatial_success_counts"
                mse=np.mean((raw[field]/(k*t)-raw["targets"])**2)
                np.testing.assert_allclose(mse,float(r["rmse_vs_target"])**2,rtol=1e-12)
    checks["matched_update_read_and_depth_accounting"]={"pairs_verified":9}
    rounded_marginal_error=0
    for r in read(root/"coupling_calibration/calibrated_fields_and_exact_moments.csv"):
        if r["field_policy"]!="moment_matched_fields":continue
        fields=np.array(json.loads(r["fields_b"]));p=np.array(json.loads(r["target_p"]))
        _,_,mean_s,_,_=exact_moments(int(r["n_spins"]),fields.astype(np.float32).astype(float),float(np.float32(r["coupling_J"])))
        rounded_marginal_error=max(rounded_marginal_error,float(np.max(np.abs((mean_s+1)/2-p))))
    checks["coupled_field_float32_rounding"]={"max_exact_marginal_error_after_rounding":rounded_marginal_error}
    assert rounded_marginal_error<1e-6
    dot=read(root/"correlated_dot/dot_product_mse.csv")
    z=[abs((float(r["empirical_mse"])-float(r["exact_full_covariance_mse"]))/float(r["empirical_mse_se"])) for r in dot]
    assert max(z)<6
    checks["correlated_dot_full_covariance"]={"conditions":len(dot),"max_abs_empirical_standard_error_units":max(z)}
    training=json.loads((root/"native_training/schedule_self_test.json").read_text())
    assert training["passed_observer_identity"] and training["passed_invalid_schedule_detector"]
    assert training["incorrect_relu_before_averaging_max_abs_difference"]>.1
    checks["training_schedule_detector"]=training
    logits=np.load(root/"training_evaluation/raw_logits.npz")
    training_rows=read(root/"training_evaluation/metrics_by_sampling_seed.csv")
    for r in training_rows:
        name=r["condition"]
        if r["representation"]=="deterministic_expected":
            key=name+"_expected_logits"
        elif r["representation"]=="UINT4_per_layer_scale_same_model":
            key=name+"_UINT4_per_layer_scale_logits"
        else:
            key=f"{name}_T{r['samples_per_layer']}_seed{r['sampling_seed']}_logits"
        assert len(logits[key])==10000
        np.testing.assert_allclose(np.mean(logits[key].argmax(1)==logits["labels"]),float(r["accuracy"]),atol=1e-12)
    checks["training_full_test_raw_accuracy"]={"conditions":len(training_rows),"examples_per_condition":10000}
    result={"scope":"Raw scalar/neural metrics, selection invariant, matched accounting, control rounding and corrected schedule self-test; not a whole-goal completion audit.","checks":checks,"passed":True}
    (root/"evidence_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
