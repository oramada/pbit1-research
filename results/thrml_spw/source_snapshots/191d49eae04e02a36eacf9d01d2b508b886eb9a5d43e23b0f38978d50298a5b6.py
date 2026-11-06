"""THRML operator accumulation with independent public-state replay on LFM3 slices."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import (AbstractObserver, SamplingSchedule, sample_states,
                   sample_with_observation, block_state_to_global, from_global_state)
from research.thrml_spw.native import NativeProbabilityBank


class Sum(AbstractObserver):
    block: object
    inputs: jax.Array
    shape: tuple
    depthwise: bool

    def init(self):return jnp.zeros((self.inputs.shape[0],self.shape[1]),jnp.float32)

    def __call__(self, program, free, clamped, carry, iteration):
        state=block_state_to_global(free+clamped,program.gibbs_spec)
        z=from_global_state(state,program.gibbs_spec,[self.block])[0].reshape(self.shape).astype(jnp.float32)
        y=jnp.einsum('bki,ki->bi',self.inputs,z) if self.depthwise else self.inputs@z
        return carry+y,None


def main():
    root=Path('results/thrml_spw/lfm3_1p7b')
    ep=root/'operator_pilot_export.npz'
    metadata=json.loads((root/'operator_pilot_export.json').read_text())
    assert hashlib.sha256(ep.read_bytes()).hexdigest()==metadata['export_sha256']
    data=np.load(ep)
    protocol={'samples':[1,16,64,256],'seeds':list(range(5)),
              'key_domain':202610290,'mapping':'per-output min/max affine; exact endpoints use explicit clamps',
              'scope':'Bounded first-layer slices, not whole-model inference; external FP32 output accumulation before nonlinearities.',
              'export_sha256':metadata['export_sha256'],
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=root/'operator_pilot';out.mkdir(exist_ok=True)
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    rows=[];raw={}
    for op,kind in enumerate(['linear','conv']):
        x=data[kind+'_inputs'];w=data[kind+'_weight']
        lower=w.min(0);span=w.max(0)-lower
        p=np.divide(w-lower,span,out=np.zeros_like(w),where=span!=0).clip(0,1)
        bank=NativeProbabilityBank(p)
        initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)]
        xj=jnp.asarray(x);isconv=kind=='conv'
        observer=Sum(bank.readout,xj,w.shape,isconv)
        for t in protocol['samples']:
            schedule=SamplingSchedule(1,t,1)
            def accumulated(key):
                total,_=sample_with_observation(key,bank.program,schedule,initial,bank.clamp_state,observer.init(),observer)
                return total
            def public(key):return sample_states(key,bank.program,schedule,initial,bank.clamp_state,[bank.readout])[0]
            acc=jax.jit(accumulated);replay=jax.jit(public)
            for seed in protocol['seeds']:
                key=jax.random.fold_in(jax.random.fold_in(jax.random.key(protocol['key_domain']),op),seed)
                total=np.asarray(acc(key));states=np.asarray(replay(key)).reshape(t,*w.shape)
                # Recompute sample outputs from public states without observer code.
                ys=[np.einsum('bki,ki->bi',x,s.astype(np.float32)) if isconv else x@s.astype(np.float32) for s in states]
                expected=np.zeros_like(total)
                for y in ys:expected+=y
                np.testing.assert_allclose(total,expected,rtol=2e-5,atol=2e-4)
                offset=x.sum(1) if isconv else x.sum(-1,keepdims=True)
                y=total/t*span+offset*lower
                name=f'{kind}_T{t}_seed{seed}'
                raw[name]=y
                rows.append({'operator':kind,'samples':t,'seed':seed,
                             'logical_weight_nodes':w.size,'free_nodes':len(bank.free_indices),
                             'retained_node_evaluations':t*len(bank.free_indices),
                             'warmup_node_evaluations':len(bank.free_indices),
                             'replay_max_abs_error':float(np.max(np.abs(total-expected))),
                             'output_mse':float(np.mean((y-data[kind+'_reference'])**2))})
            print(kind,t,'all seeds and public replay passed',flush=True)
    np.savez_compressed(out/'outputs.npz',**raw)
    (out/'results.json').write_text(json.dumps({'cases':rows,'all_public_replays_passed':True,
        'scope':protocol['scope'],'outputs_sha256':hashlib.sha256((out/'outputs.npz').read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
