"""Native THRML p-bit inference on a learned nonnegative-weight MNIST model.

Every neural matrix weight is its corresponding node's probability, without an
affine range mapping. Each layer averages sampled linear outputs before ReLU.
The arithmetic is a modeled JAX observer, not native physical TSU arithmetic.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from thrml import (AbstractObserver, SamplingSchedule, block_state_to_global,
                   from_global_state, sample_with_observation)

from .native import NativeProbabilityBank
from .provenance import manifest


class PreactivationSum(AbstractObserver):
    readout: object
    inputs: jax.Array
    weight_shape: tuple

    def init(self):return jnp.zeros((self.inputs.shape[0],self.weight_shape[1]),dtype=jnp.float32)

    def __call__(self,program,state_free,state_clamped,carry,iteration):
        global_state=block_state_to_global(state_free+state_clamped,program.gibbs_spec)
        z=from_global_state(global_state,program.gibbs_spec,[self.readout])[0]
        return carry+self.inputs@z.reshape(self.weight_shape).astype(jnp.float32),None


class NativeLinear:
    def __init__(self,weight,bias):
        self.bank=NativeProbabilityBank(weight)
        self.bias=jnp.asarray(bias)

    @lru_cache(maxsize=8)
    def compiled(self,t):
        bank=self.bank
        schedule=SamplingSchedule(1,t,1)
        initial=[jnp.zeros((len(bank.free_indices),),dtype=jnp.bool_)] if len(bank.free_indices) else []
        def run(key,inputs):
            observer=PreactivationSum(bank.readout,inputs,bank.shape)
            total,_=sample_with_observation(key,bank.program,schedule,initial,
                          bank.clamp_state,observer.init(),observer)
            return total/t+self.bias
        return jax.jit(run)


def deterministic(x,weights,biases):
    return np.maximum(x@weights[0]+biases[0],0)@weights[1]+biases[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(5)))
    parser.add_argument("--samples",type=int,nargs="+",default=[1,4,16,64,256])
    parser.add_argument("--examples",type=int,default=1024)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--out",default="results/thrml_spw/positive_inference")
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=Path("data/thrml_spw/positive_mnist")
    evaluation=np.load(source/"evaluation_data.npz")
    ids=np.sort(np.random.default_rng(20261027).choice(len(evaluation["test_labels"]),args.examples,replace=False))
    x,y=evaluation["test_inputs"][ids],evaluation["test_labels"][ids]
    rows=[];controls={};raw={"example_indices":ids,"labels":y};start=time.monotonic()
    max_export_error=0
    for seed in args.seeds:
        exported=np.load(source/f"seed{seed}.npz")
        weights=[exported[f"weight_{i}"] for i in range(2)]
        biases=[exported[f"bias_{i}"] for i in range(2)]
        reference=deterministic(x,weights,biases)
        export_error=float(np.max(np.abs(reference-exported["reference_test_logits"][ids])))
        max_export_error=max(max_export_error,export_error)
        if export_error>1e-4:raise RuntimeError("Exported model differs from training forward")
        baseline=float(np.mean(reference.argmax(1)==y))
        def record(representation,t,logits,elapsed,updates=0):
            rows.append({"seed":seed,"representation":representation,"samples_per_layer":t,
                "examples":len(y),"accuracy":float(np.mean(logits.argmax(1)==y)),
                "baseline_accuracy":baseline,"accuracy_delta":float(np.mean(logits.argmax(1)==y))-baseline,
                "cross_entropy":float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])),
                "logit_rmse":float(np.sqrt(np.mean((logits-reference)**2))),
                "logical_matrix_weight_pbits":sum(w.size for w in weights) if t else 0,
                "deterministic_neural_biases":sum(b.size for b in biases),
                "executed_free_node_gibbs_updates":updates,"simulator_wall_seconds":elapsed})
            raw[f"seed{seed}_{representation}_T{t}"]=logits
        record("FP32",0,reference,0)
        fp16=deterministic(x,[w.astype(np.float16).astype(np.float32) for w in weights],
                           [b.astype(np.float16).astype(np.float32) for b in biases])
        record("FP16_rounded_values",0,fp16,0)
        for bits in (4,8):
            quant=[]
            for w in weights:
                scale=float(w.max())/((1<<bits)-1)
                quant.append(np.rint(w/scale).clip(0,(1<<bits)-1)*scale if scale else w)
            record(f"UINT{bits}_per_layer_scale",0,deterministic(x,quant,biases),0)
        layers=[NativeLinear(w,b) for w,b in zip(weights,biases)]
        controls[str(seed)]=[layer.bank.controls() for layer in layers]
        for t in args.samples:
            begin=time.monotonic();logits=[]
            for batch,start_i in enumerate(range(0,len(x),args.batch_size)):
                xb=jnp.asarray(x[start_i:start_i+args.batch_size])
                # Independent layer/batch keys; identical seed policy across budgets.
                key=jax.random.fold_in(jax.random.key(seed+73000),batch)
                k0,k1=jax.random.split(key)
                hidden=jax.nn.relu(layers[0].compiled(t)(k0,xb))
                logits.append(np.asarray(layers[1].compiled(t)(k1,hidden)))
            logits=np.concatenate(logits)
            updates=int(t*len(logits[::args.batch_size])*sum(len(layer.bank.free_indices) for layer in layers))
            record("THRML_layer_preactivation",t,logits,time.monotonic()-begin,updates)
            print(f"seed={seed} T={t} accuracy={rows[-1]['accuracy']:.4f} FP={baseline:.4f}",flush=True)
    np.savez_compressed(out/"predictions.npz",**raw)
    with (out/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for rep,t in sorted({(r["representation"],r["samples_per_layer"]) for r in rows}):
        group=[r for r in rows if r["representation"]==rep and r["samples_per_layer"]==t]
        a=np.array([r["accuracy"] for r in group]);delta=np.array([r["accuracy_delta"] for r in group])
        ci=1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else float("nan")
        summary.append({"representation":rep,"samples_per_layer":t,"seeds":len(a),
            "accuracy_mean":float(a.mean()),"accuracy_ci95_margin":float(ci),
            "paired_accuracy_delta_mean":float(delta.mean())})
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    result={"elapsed_seconds":time.monotonic()-start,"controls_by_model":controls,
            "max_training_export_logit_error":max_export_error,
            "sample_trajectories_materialized":False,"float_weight_reconstruction_buffer":False,
            "probability_mapping":"direct w=p; no signed affine mapping",
            "execution":"THRML Gibbs states; layer output sums in JAX observer before ReLU",
            "arithmetic_scope":"CPU simulation of external matrix/activation arithmetic, not a native hardware gate claim",
            "counter_interpretation":"each sampled bank state shared across the batch; reported updates are per evaluated dataset, not hardware inference-energy estimates",
            "summary":summary}
    manifest(out/"manifest.json",vars(args),result)
    print(summary,flush=True)


if __name__=="__main__":main()
