"""Task-level logical resource table from measured native THRML inference.

This is accounting, not a performance simulator. A p-bit cell, a digital storage
bit and an analog controller are not assigned interchangeable areas or energies.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .provenance import manifest


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    root=Path("results/thrml_spw")
    original=read(root/"positive_full_test/metrics.csv")
    controlled=read(root/"control_neural/test_metrics.csv")
    provenance=json.loads((root/"positive_full_test/manifest.json").read_text())
    bank_info=provenance["result"]["controls_by_model"]
    rows=[];n_weights=2368;bias_bits=42*32;layers=2;batch=32
    for r in original:
        seed=int(r["seed"]);rep=r["representation"];t=int(r["samples_per_layer"])
        if rep not in ("FP32","FP16_rounded_values","UINT4_per_layer_scale","UINT8_per_layer_scale","THRML_layer_preactivation"):
            continue
        free=sum(c["free_pbits"] for c in bank_info[str(seed)])
        if t:
            controls=32*free;mode_bits=2*n_weights;shared=layers*32
            digital_weight_bits=0
        else:
            controls=mode_bits=shared=0
            storage_width={"FP32":32,"FP16_rounded_values":16,"UINT4_per_layer_scale":4,"UINT8_per_layer_scale":8}[rep]
            digital_weight_bits=n_weights*storage_width+(layers*32 if rep.startswith("UINT") else 0)
        rows.append({"seed":seed,"representation":rep,"samples_per_layer":t,
            "accuracy":float(r["accuracy"]),"logical_pbit_cells":n_weights if t else 0,
            "free_pbits":free if t else 0,"endpoint_clamped_cells":n_weights-free if t else 0,
            "ideal_single_example_free_updates":free*t if t else 0,
            "ideal_single_example_weight_readouts":n_weights*t if t else 0,
            "sequential_layer_sampler_sweeps":layers*t if t else 0,
            "modeled_binary_weight_product_terms_per_example":n_weights*t if t else 0,
            "ordinary_dense_product_terms_per_example":n_weights if not t else 0,
            "hypothetical_digital_weight_storage_bits":digital_weight_bits,
            "hypothetical_bias_field_code_bits":controls,
            "hypothetical_endpoint_mode_bits":mode_bits,
            "hypothetical_shared_control_bits":shared,
            "deterministic_neural_bias_bits":bias_bits,
            "hypothetical_weight_related_digital_bits":digital_weight_bits+controls+mode_bits+shared+bias_bits,
            "modeled_largest_native_output_accumulator_bits_batch32":32*32*32 if t else 0,
            "modeled_largest_native_output_accumulator_bits_batch1":32*32 if t else 0})
    for r in controlled:
        t=int(r["samples_per_layer"]);free=int(r["free_nodes"])
        field=int(r["hypothetical_packed_field_code_bits"])
        mode=int(r["hypothetical_endpoint_mode_bits"])
        shared=int(r["hypothetical_shared_range_beta_bits"])
        rows.append({"seed":int(r["seed"]),"representation":"THRML_selected_6bit_control",
            "samples_per_layer":t,"accuracy":float(r["accuracy"]),
            "logical_pbit_cells":n_weights,"free_pbits":free,"endpoint_clamped_cells":n_weights-free,
            "ideal_single_example_free_updates":free*t,
            "ideal_single_example_weight_readouts":n_weights*t,
            "sequential_layer_sampler_sweeps":layers*t,
            "modeled_binary_weight_product_terms_per_example":n_weights*t,
            "ordinary_dense_product_terms_per_example":0,"hypothetical_digital_weight_storage_bits":0,
            "hypothetical_bias_field_code_bits":field,"hypothetical_endpoint_mode_bits":mode,
            "hypothetical_shared_control_bits":shared,"deterministic_neural_bias_bits":bias_bits,
            "hypothetical_weight_related_digital_bits":field+mode+shared+bias_bits,
            "modeled_largest_native_output_accumulator_bits_batch32":32*32*32,
            "modeled_largest_native_output_accumulator_bits_batch1":32*32})
    out=root/"resource_frontier";out.mkdir(parents=True,exist_ok=True)
    with (out/"per_checkpoint.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    numeric=[k for k in rows[0] if k not in ("seed","representation","samples_per_layer")]
    for rep,t in sorted({(r["representation"],r["samples_per_layer"]) for r in rows}):
        group=[r for r in rows if r["representation"]==rep and r["samples_per_layer"]==t]
        row={"representation":rep,"samples_per_layer":t,"checkpoints":len(group)}
        for field in numeric:row[field+"_mean"]=float(np.mean([r[field] for r in group]))
        summary.append(row)
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/"assumptions.json").write_text(json.dumps({
        "purpose":"Logical resource accounting from measured inference, not predicted energy or area",
        "neural_weight_count":n_weights,"bias_parameters":42,"layers":layers,
        "native_hardware_multiply_and_activation":"not implemented; JAX observer arithmetic",
        "sampler_semantics":"warmup=1 emits first state; T observations use exactly T updates per free node",
        "parallelism":"all free nodes of a bank update in one logical Gibbs sweep; two layers depend sequentially",
        "static_programming":"bias controls programmed once and reused; not charged as writes per inference",
        "simulated_batch_sharing":"one sampled bank state serves 32 inputs; actual dataset updates are separately in run metrics",
        "single_example_counts":"isolated logical inference without reusing samples across other examples",
        "accumulator_bits":"float32 output accumulator for the largest 32-unit layer; input/hidden tensors, transient sampled operands, routing and workspaces are additional",
        "control_packing":"B bits per free field +2 mode bits per logical node; explicit hypothetical design, not a lower bound",
        "endpoint_optimization":"all weight cells counted, including clamps; no sparsity compression assumed for any dense representation",
        "numeric_baseline":"UINT4/8 use quantized values executed with conventional float arithmetic; no optimized low-bit accelerator measured",
        "unknown_physical_terms":["p-bit cell area","retained bias controller area/precision","readout cost","binary gating or multiply cost","accumulation cost","clock/communication cost","device switching/mixing rate"]},indent=2)+"\n")
    paths=[root/"positive_full_test/metrics.csv",root/"positive_full_test/manifest.json",
           root/"control_neural/test_metrics.csv"]
    manifest(out/"manifest.json",{"input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "batch_size":batch,"weight_count":n_weights,"neural_bias_bits":bias_bits},
        {"per_checkpoint_rows":len(rows),"summary_rows":len(summary),
         "scope":"Measured task quality with explicit logical accounting; no physical area, energy, latency or bit-cell equivalence asserted."})
    print(out/"summary.csv")


if __name__=="__main__":main()
