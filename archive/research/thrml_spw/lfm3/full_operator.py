"""Bounded-bank THRML execution of full LFM3 operators, with exact state replay.

Tiling time-multiplexes simulator nodes; logical model weight coordinates are
distinct from maximum concurrently instantiated simulator nodes.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import hashlib
import json
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_states,sample_with_observation
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
from research.thrml_spw.lfm3.native_operator_pilot_v2 import Sum


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'full_operator';out.mkdir(exist_ok=True)
    ep=root/'full_operator_export.npz'
    assert sha(ep)==json.loads((root/'full_operator_export.json').read_text())['export_sha256']
    protocol={'source_sha256':sha(Path(__file__)),'export_sha256':sha(ep),
              'samples':[1,16,64],'seeds':[0],'tile_output_channels':64,
              'field_bits':8,'field_range':[-4,4],'input_format':'signed ACT8 per operator tensor',
              'accumulator':'int32, checked bound; replay NumPy int64 exact',
              'key_domain':202610291,'scope':'Full first-layer operators separately; no full stochastic model or perplexity claim. Physical sampling, reprogramming and readout costs unmeasured.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    data=np.load(ep);rows=[];outputs={};begin=time.monotonic()
    for op,kind in enumerate(['linear','conv']):
        xf=data[kind+'_inputs'];w=data[kind+'_weight'];conv=kind=='conv'
        scale=float(np.max(np.abs(xf)))/127 if np.any(xf) else 1.0
        x=np.rint(xf/scale).clip(-127,127).astype(np.int32)
        bank=NativeProbabilityBank(np.full((w.shape[0],64),.5))
        initial=[jnp.zeros(bank.logical_pbits,dtype=jnp.bool_)]
        assert 127*w.shape[0]*64 < np.iinfo(np.int32).max
        def reference(inputs,weights):
            return np.einsum('bki,ki->bi',inputs,weights) if conv else inputs@weights
        exact=reference(xf.astype(np.float64),w.astype(np.float64))
        quant=reference(x.astype(np.float64)*scale,w.astype(np.float64))
        for t in protocol['samples']:
            schedule=SamplingSchedule(1,t,1)
            def accumulated(key,inputs,fields):
                program=set_fields(bank,fields)
                obs=Sum(bank.readout,inputs,bank.shape,conv)
                total,_=sample_with_observation(key,program,schedule,initial,[],obs.init(),obs)
                return total
            def public(key,fields):
                return sample_states(key,set_fields(bank,fields),schedule,initial,[],[bank.readout])[0]
            acc=jax.jit(accumulated);replay=jax.jit(public)
            ys=[];means=[];codes_all=[];start=time.monotonic()
            for tile,first in enumerate(range(0,w.shape[1],64)):
                wt=w[:,first:first+64];xt=x[:,:,first:first+64] if conv else x
                lo=wt.min(0).astype(np.float64);span=wt.max(0).astype(np.float64)-lo
                p=np.divide(wt-lo,span,out=np.full_like(wt,.5,dtype=np.float64),where=span!=0)
                pf=np.clip(p,1e-12,1-1e-12)
                field=np.clip((np.log(pf)-np.log1p(-pf))/2,-4,4)
                codes=np.rint((field+4)*255/8).clip(0,255).astype(np.uint8)
                fields=(-4+codes.astype(np.float32)*np.float32(8/255)).reshape(-1)
                realized=1/(1+np.exp(-2*fields.astype(np.float64))).reshape(wt.shape)
                key=jax.random.fold_in(jax.random.fold_in(jax.random.key(protocol['key_domain']),op),tile)
                total=np.asarray(acc(key,jnp.asarray(xt),jnp.asarray(fields)))
                states=np.asarray(replay(key,jnp.asarray(fields))).reshape(t,*wt.shape)
                check=np.zeros_like(total,dtype=np.int64)
                for state in states:check+=reference(xt.astype(np.int64),state.astype(np.int64))
                np.testing.assert_array_equal(total.astype(np.int64),check)
                offset=xt.sum(1) if conv else xt.sum(-1,keepdims=True)
                ys.append((total.astype(np.float64)/t*span+offset*lo)*scale)
                means.append(reference(xt.astype(np.float64)*scale,lo+span*realized))
                codes_all.append(codes)
                (out/'progress.json').write_text(json.dumps({'operator':kind,'samples':t,'tiles_completed':tile+1,'tiles_total':w.shape[1]//64,'elapsed_seconds':time.monotonic()-begin})+'\n')
            y=np.concatenate(ys,axis=1);mean=np.concatenate(means,axis=1)
            outputs[f'{kind}_T{t}']=y;outputs[f'{kind}_nominal_mean']=mean
            outputs[f'{kind}_field_codes']=np.concatenate(codes_all,axis=1)
            row={'operator':kind,'samples':t,'seed':0,'weights_executed':w.size,
                 'tiles':len(ys),'max_concurrent_weight_nodes':bank.logical_pbits,
                 'retained_node_evaluations_per_pass':int(w.size*t),
                 'warmup_node_evaluations_per_pass':int(w.size),
                 'all_integer_public_replays_exact':True,
                 'output_MSE':float(np.mean((y-exact)**2)),
                 'ACT8_floor_MSE':float(np.mean((quant-exact)**2)),
                 'nominal_control_floor_MSE':float(np.mean((mean-exact)**2)),
                 'elapsed_seconds_including_replay':time.monotonic()-start}
            rows.append(row);print(json.dumps(row),flush=True)
        del bank
    np.savez_compressed(out/'outputs.npz',**outputs)
    result={'cases':rows,'outputs_sha256':sha(out/'outputs.npz'),
            'elapsed_seconds':time.monotonic()-begin,
            'process_max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'devices':[str(d) for d in jax.devices()],
            'scope':protocol['scope'],'public_replay_work_charged_separately':True}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
