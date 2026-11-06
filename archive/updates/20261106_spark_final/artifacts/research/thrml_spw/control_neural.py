"""Choose finite bias precision on validation data; run native THRML on test data.

The field quantizer is selected without looking at test labels. Native inference
then uses the same seeds, ordered test examples, and layer schedule as the fixed
float32-control reference. Quantized controls do not replace THRML sampling.
"""
from __future__ import annotations

import csv
import json
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from .control_bank import ControlledProbabilityBank
from .positive_inference import NativeLinear, deterministic
from .provenance import manifest


def ce(logits,labels):
    return float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels]))


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class ControlledLinear(NativeLinear):
    def __init__(self,weight,bias,bits,bound):
        self.bank=ControlledProbabilityBank(weight,bits,bound)
        self.bias=jnp.asarray(bias)


def main():
    out=Path("results/thrml_spw/control_neural");out.mkdir(parents=True,exist_ok=True)
    source=Path("data/thrml_spw/positive_mnist")
    data=np.load(source/"evaluation_data.npz")
    xv,yv=data["validation_inputs"],data["validation_labels"]
    x,y=data["test_inputs"],data["test_labels"]
    seeds=list(range(5));precision=[4,6,8,10,12];bounds=[2.,4.,8.];budgets=[64,256]
    models={};profiles=[];baseline_validation=[];start=time.monotonic()
    for seed in seeds:
        z=np.load(source/f"seed{seed}.npz")
        weights=[z[f"weight_{i}"] for i in range(2)];biases=[z[f"bias_{i}"] for i in range(2)]
        models[seed]=(weights,biases)
        reference=deterministic(xv,weights,biases)
        baseline_validation.append({"seed":seed,"accuracy":float(np.mean(reference.argmax(1)==yv)),"cross_entropy":ce(reference,yv)})
        for bits in precision:
            for bound in bounds:
                banks=[ControlledProbabilityBank(w,bits,bound) for w in weights]
                realized=[bank.realized_probabilities.astype(np.float32) for bank in banks]
                logits=deterministic(xv,realized,biases)
                profiles.append({"seed":seed,"field_bits":bits,"field_bound":bound,
                    "validation_accuracy":float(np.mean(logits.argmax(1)==yv)),
                    "validation_cross_entropy":ce(logits,yv),
                    "matrix_weight_control_rmse":float(np.sqrt(np.mean(np.concatenate([(q-w).ravel()**2 for q,w in zip(realized,weights)])))),
                    "clipped_fields":sum(bank.clipped_fields for bank in banks)})
    write_csv(out/"validation_profiles.csv",profiles)
    base_accuracy=float(np.mean([r["accuracy"] for r in baseline_validation]))
    base_ce=float(np.mean([r["cross_entropy"] for r in baseline_validation]))
    choices=[]
    for bits in precision:
        by_bound=[]
        for bound in bounds:
            group=[r for r in profiles if r["field_bits"]==bits and r["field_bound"]==bound]
            by_bound.append({"field_bits":bits,"field_bound":bound,
                "mean_validation_accuracy":float(np.mean([r["validation_accuracy"] for r in group])),
                "mean_validation_cross_entropy":float(np.mean([r["validation_cross_entropy"] for r in group]))})
        best=min(by_bound,key=lambda r:(r["mean_validation_cross_entropy"],r["field_bound"]))
        best["accuracy_drop"]=base_accuracy-best["mean_validation_accuracy"]
        best["cross_entropy_increase"]=best["mean_validation_cross_entropy"]-base_ce
        best["passes_predeclared_control_tolerance"]=(best["accuracy_drop"]<=.005 and best["cross_entropy_increase"]<=.02)
        choices.append(best)
    passing=[r for r in choices if r["passes_predeclared_control_tolerance"]]
    selected=min(passing,key=lambda r:r["field_bits"]) if passing else choices[-1]
    bits,bound=selected["field_bits"],selected["field_bound"]
    selection={"rule":"For each bit width choose the range with lowest mean validation CE; then choose smallest B with mean accuracy drop <=0.005 and CE increase <=0.02.",
        "validation_examples":len(yv),"test_labels_used_in_selection":False,
        "baseline_validation_by_seed":baseline_validation,"choices":choices,"selected":selected}
    (out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print("Selected",selected,flush=True)
    reference_rows=list(csv.DictReader(Path("results/thrml_spw/positive_full_test/metrics.csv").open()))
    rows=[];raw={"labels":y};controls={};n_batches=(len(x)+31)//32
    for seed in seeds:
        weights,biases=models[seed]
        layers=[ControlledLinear(w,b,bits,bound) for w,b in zip(weights,biases)]
        controls[str(seed)]=[layer.bank.controls() for layer in layers]
        expected=deterministic(x,[layer.bank.realized_probabilities.astype(np.float32) for layer in layers],biases)
        raw[f"seed{seed}_control_mean_logits"]=expected
        fp=deterministic(x,weights,biases);fp_accuracy=float(np.mean(fp.argmax(1)==y))
        for t in budgets:
            begin=time.monotonic();outputs=[]
            for batch,first in enumerate(range(0,len(x),32)):
                key=jax.random.fold_in(jax.random.key(seed+73000),batch)
                k0,k1=jax.random.split(key)
                hidden=jax.nn.relu(layers[0].compiled(t)(k0,jnp.asarray(x[first:first+32])))
                outputs.append(np.asarray(layers[1].compiled(t)(k1,hidden)))
            logits=np.concatenate(outputs);raw[f"seed{seed}_T{t}"]=logits
            ideal=next(r for r in reference_rows if int(r["seed"])==seed and
                       r["representation"]=="THRML_layer_preactivation" and int(r["samples_per_layer"])==t)
            free=sum(len(layer.bank.free_indices) for layer in layers)
            nweights=sum(w.size for w in weights)
            rows.append({"seed":seed,"field_bits":bits,"field_bound":bound,"samples_per_layer":t,
                "test_examples":len(y),"accuracy":float(np.mean(logits.argmax(1)==y)),
                "cross_entropy":ce(logits,y),"FP32_model_accuracy":fp_accuracy,
                "float32_control_native_accuracy":float(ideal["accuracy"]),
                "accuracy_delta_vs_float32_control_native":float(np.mean(logits.argmax(1)==y))-float(ideal["accuracy"]),
                "quantized_control_expected_network_accuracy":float(np.mean(expected.argmax(1)==y)),
                "logical_weight_nodes":nweights,"free_nodes":free,"clamped_nodes":nweights-free,
                "executed_free_node_updates":t*free*n_batches,
                "hypothetical_packed_field_code_bits":free*bits,
                "hypothetical_endpoint_mode_bits":2*nweights,
                "hypothetical_shared_range_beta_bits":2*64,
                "deterministic_neural_bias_bits":42*32,
                "UINT4_weight_scale_and_neural_bias_bits":nweights*4+2*32+42*32,
                "simulator_wall_seconds":time.monotonic()-begin})
            print("seed",seed,"T",t,"native quantized",rows[-1]["accuracy"],"ideal control",ideal["accuracy"],flush=True)
    write_csv(out/"test_metrics.csv",rows)
    np.savez_compressed(out/"predictions.npz",**raw)
    summary=[]
    for t in budgets:
        group=[r for r in rows if r["samples_per_layer"]==t]
        a=np.array([r["accuracy"] for r in group]);d=np.array([r["accuracy_delta_vs_float32_control_native"] for r in group])
        summary.append({"samples_per_layer":t,"models":len(a),"accuracy_mean":float(a.mean()),
            "accuracy_ci95_margin":float(1.96*a.std(ddof=1)/np.sqrt(len(a))),
            "paired_control_delta_mean":float(d.mean()),
            "paired_control_delta_ci95_margin":float(1.96*d.std(ddof=1)/np.sqrt(len(d)))})
    write_csv(out/"summary.csv",summary)
    manifest(out/"manifest.json",{"precision_bits":precision,"field_bounds":bounds,"seeds":seeds,
        "sample_budgets":budgets,"selection_rule":selection["rule"],"selected":selected},
        {"validation_selection":selection,"controls_by_seed":controls,"summary":summary,
         "elapsed_seconds":time.monotonic()-start,
         "scope":"Dimensionless simulated field quantization; packed digital control counts are an explicit accounting assumption, not measured controller area."})


if __name__=="__main__":main()
