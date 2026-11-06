"""Accumulate sampled linear outputs inside THRML's observer, then activate.

All random bit states come from THRML. Matrix multiplication and ReLU are modeled
JAX arithmetic in an observer, not claims of native TSU multiply/activation gates.
The direct accumulator retains output sums, not reconstructed individual weights.
"""
from __future__ import annotations

import csv
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from thrml import (AbstractObserver, SamplingSchedule, block_state_to_global,
                   from_global_state, sample_with_observation)

from .native import NativeProbabilityBank
from .provenance import manifest


class LinearObserver(AbstractObserver):
    readout: object
    inputs: jax.Array
    bias: jax.Array
    weight_shape: tuple

    def init(self):
        shape=(self.inputs.shape[0],self.weight_shape[1])
        return (jnp.zeros(shape,dtype=jnp.float32),jnp.zeros(shape,dtype=jnp.float32))

    def __call__(self,program,state_free,state_clamped,carry,iteration):
        global_state=block_state_to_global(state_free+state_clamped,program.gibbs_spec)
        weight=from_global_state(global_state,program.gibbs_spec,[self.readout])[0]
        y=self.inputs@weight.reshape(self.weight_shape).astype(jnp.float32)
        return (carry[0]+y,carry[1]+jax.nn.relu(y+self.bias)),None


class WeightObserver(AbstractObserver):
    readout: object
    size: int

    def init(self):return jnp.zeros(self.size,dtype=jnp.float32)

    def __call__(self,program,state_free,state_clamped,carry,iteration):
        global_state=block_state_to_global(state_free+state_clamped,program.gibbs_spec)
        bits=from_global_state(global_state,program.gibbs_spec,[self.readout])[0]
        return carry+bits.astype(jnp.float32),None


def observe(bank,observer,seed,samples,chains):
    schedule=SamplingSchedule(1,samples,1)
    initial=[jnp.zeros((bank.free_indices.size,),dtype=jnp.bool_)] if bank.free_indices.size else []
    def one(key):
        carry,records=sample_with_observation(key,bank.program,schedule,initial,
                             bank.clamp_state,observer.init(),observer)
        return carry
    keys=jax.random.split(jax.random.key(seed),chains)
    return jax.device_get(jax.jit(jax.vmap(one))(keys))


def main():
    out=Path("results/thrml_spw/preactivation");out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(730)
    weight=rng.uniform(.02,.98,size=(16,8)).astype(np.float32)
    inputs=(rng.normal(size=(8,16))/4).astype(np.float32)
    bias=rng.uniform(-.5,.2,size=8).astype(np.float32)
    bank=NativeProbabilityBank(weight)
    direct=LinearObserver(bank.readout,jnp.asarray(inputs),jnp.asarray(bias),weight.shape)
    reconstruct=WeightObserver(bank.readout,weight.size)
    reference=inputs@weight+bias
    nonlinear_reference=np.maximum(reference,0)
    variance=float(np.mean((inputs**2)@(weight*(1-weight))))
    rows=[];max_identity=0;max_activation_identity=0;start=time.monotonic()
    budgets=[1,4,16,64,256,1024,4096];seeds=list(range(5));chains=32
    raw={}
    for samples in budgets:
        for seed in seeds:
            # Both observers see exactly the same Gibbs paths; their state updates
            # are identical and only the observer's retained carry differs.
            ysum,after_sum=observe(bank,direct,seed,samples,chains)
            weight_sum=observe(bank,reconstruct,seed,samples,chains)
            pre=ysum/samples+bias
            from_weights=inputs@(weight_sum.reshape((chains,*weight.shape))/samples)+bias
            before=np.maximum(pre,0);after=after_sum/samples
            identity=float(np.max(np.abs(pre-from_weights)))
            activation_identity=float(np.max(np.abs(before-np.maximum(from_weights,0))))
            max_identity=max(max_identity,identity);max_activation_identity=max(max_activation_identity,activation_identity)
            rows.append({"seed":seed,"samples":samples,"trials":chains,
                "linear_rmse":float(np.sqrt(np.mean((pre-reference)**2))),
                "theory_linear_rmse":float(np.sqrt(variance/samples)),
                "relu_after_accumulation_rmse":float(np.sqrt(np.mean((before-nonlinear_reference)**2))),
                "average_relu_outputs_rmse":float(np.sqrt(np.mean((after-nonlinear_reference)**2))),
                "linear_identity_max_error":identity,"relu_identity_max_error":activation_identity})
            raw[f"seed{seed}_T{samples}_pre"]=pre
            raw[f"seed{seed}_T{samples}_after"]=after
        print(f"completed T={samples} elapsed={time.monotonic()-start:.1f}s",flush=True)
    np.savez_compressed(out/"outputs.npz",weights=weight,inputs=inputs,bias=bias,**raw)
    with (out/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={"max_linear_identity_error":max_identity,
            "max_activation_identity_error":max_activation_identity,
            "identity_tolerance_float32":0.0001,"passed":max_identity<.0001,
            "controls":bank.controls(),"direct_accumulator_floats":int(2*reference.size),
            "production_pre_activation_only_accumulator_floats":int(reference.size),
            "reconstructed_weight_buffer_floats":int(weight.size),
            "sampler_retains_sample_trajectory":False,
            "arithmetic_scope":"JAX observer on simulator CPU; no TSU multiplier or activation claimed",
            "elapsed_seconds":time.monotonic()-start}
    manifest(out/"manifest.json",{"weights_shape":weight.shape,"inputs_shape":inputs.shape,
             "seeds":seeds,"chains_per_seed":chains,"budgets":budgets,
             "matrix_and_activation_dtype":"float32"},result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
    for field,label in (("relu_after_accumulation_rmse","ReLU after averaging linear outputs"),
                        ("average_relu_outputs_rmse","Average ReLU outputs")):
        ax.loglog(budgets,[np.mean([r[field] for r in rows if r["samples"]==t]) for t in budgets],"o-",label=label)
    ax.set(xlabel="THRML samples per weight",ylabel="Output RMSE",title="Positive float weights encoded as p-bit probabilities")
    ax.legend();fig.savefig(out/"preactivation.png",dpi=180);fig.savefig(out/"preactivation.pdf")
    print(result,flush=True)
    if not result["passed"]:raise RuntimeError("Preactivation identity check failed")


if __name__=="__main__":main()
