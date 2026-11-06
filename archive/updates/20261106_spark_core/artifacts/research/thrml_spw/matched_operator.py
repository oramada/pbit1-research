"""Actual large-operator spatial-replica/time comparison at equal update budgets.

This is a replica control, explicitly not positional bit encoding or an official
Extropic learned-weight reference. It reads only old development activations.
"""
import json,csv,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .packed_native import PackedNativeLinear
from .replicated_native import ReplicatedNativeLinear
from .packed_controls import load_controls,expected_weights
from .verify_transformer_control import sha
from .provenance import manifest


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA required for this pinned operator study')
    out=Path('results/thrml_spw/matched_operator');out.mkdir(exist_ok=True)
    artifact=Path('results/thrml_spw/packed_quarter_controls/stage11.npz');source=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp,rp=[source/f'stage11_{n}.npy' for n in ['inputs','conditional_reference']];x,ref=np.load(xp),np.load(rp);loaded=load_controls(artifact)
    protocol={'replica_conditions':{'temporal':{'K':1,'T':512},'spatial':{'K':4,'T':128}},'seeds':list(range(10)),'contexts_per_bank':8,'domain':1020730,
        'key_schedule':'fold condition_index, seed, stage11, batch into domain key; disjoint streams between modes, not identical trajectories',
        'artifact_sha256':sha(artifact),'inputs_sha256':sha(xp),'reference_sha256':sha(rp),'source_sha256':sha(__file__),
        'scope':'Same fixed actual old student activation matrix and code-programmed expectations. Equal stochastic-device evaluations and logical read totals. Not new-study text, full neural quality, positional K-bit encoding or official Extropic model performance.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    mean=x.astype(np.float64)@expected_weights(loaded)+loaded['bias'];bias=float(np.mean((mean-ref)**2))
    shape=tuple(loaded['header']['shape']);span=np.repeat(loaded['span'],128,axis=0)[:shape[0]]
    q=loaded['probabilities'];variance=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@(span**2*q*(1-q)))/512)
    rows=[];controls={};raw={};started=time.monotonic()
    for condition_index,(name,k,t) in enumerate([('temporal',1,512),('spatial',4,128)]):
        layer=PackedNativeLinear(artifact) if k==1 else ReplicatedNativeLinear(artifact,k)
        controls[name]=layer.controls();bank=layer.bank;fn=layer.compiled(t)
        for seed in range(10):
            chunks=[]
            for batch,first in enumerate(range(0,len(x),8)):
                key=jax.random.key(1020730)
                for value in [condition_index,seed,11,batch]:key=jax.random.fold_in(key,value)
                xb=x[first:first+8];y=fn(key,jnp.asarray(xb.reshape(-1,xb.shape[-1])))
                chunks.append(np.asarray(y).reshape(xb.shape[0],xb.shape[1],-1))
            y=np.concatenate(chunks);raw[f'{name}_seed{seed}']=y
            mse=float(np.mean((y.astype(np.float64)-ref)**2));centered=float(np.mean((y.astype(np.float64)-mean)**2))
            batches=(len(x)+7)//8
            rows.append({'condition':name,'seed':seed,'K':k,'T':t,'total_draws_per_unique_weight_per_bank':k*t,
                'unique_weights':int(np.prod(shape)),'logical_nodes':bank.logical_pbits,'free_nodes':len(bank.free_indices),
                'bank_batches':batches,'parallel_steps_per_bank':t,'free_node_evaluations':batches*t*len(bank.free_indices),
                'logical_weight_readouts':batches*t*bank.logical_pbits,'nominal_binary_product_terms':int(x.shape[0]*x.shape[1]*bank.logical_pbits*t),
                'actual_float32_field_bytes':bank.biases.nbytes,'conditional_mse':mse,'centered_sampling_mse':centered,
                'predicted_sampling_mse':variance,'control_bias_mse':bias,'predicted_total_mse':variance+bias})
            print(f'Matched operator {name} K{k} T{t} seed{seed}: MSE={mse:.6g} elapsed={time.monotonic()-started:.1f}s',flush=True)
    assert len(rows)==20
    for metric in ['free_node_evaluations','logical_weight_readouts','nominal_binary_product_terms']:
        assert len({r[metric] for r in rows})==1
    summaries=[]
    for name in ['temporal','spatial']:
        group=[r for r in rows if r['condition']==name];values=np.array([r['conditional_mse'] for r in group]);centered=np.array([r['centered_sampling_mse'] for r in group])
        summaries.append({'condition':name,'seeds':10,'MSE_mean':float(values.mean()),'MSE_mc95_margin':float(1.96*values.std(ddof=1)/np.sqrt(10)),
            'sampling_MSE_mean':float(centered.mean()),'sampling_MSE_over_prediction':float(centered.mean()/variance),'predicted_total_MSE':variance+bias})
    differences=np.array([rows[10+i]['conditional_mse']-rows[i]['conditional_mse'] for i in range(10)])
    for filename,items in [('metrics.csv',rows),('summary.csv',summaries)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    np.savez_compressed(out/'outputs.npz',**raw)
    result={'conditions':controls,'summaries':summaries,'spatial_minus_temporal_MSE_mean':float(differences.mean()),
        'spatial_minus_temporal_MC95_margin':float(1.96*differences.std(ddof=1)/np.sqrt(10)),
        'elapsed_seconds_including_init_and_audit':time.monotonic()-started,
        'scope':protocol['scope']+' Logical parallel depths are not measured device latency. Replicated float32 fields are actual simulator overhead; physical bias-sharing circuitry is not modeled.'}
    (out/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(out/'manifest.json',protocol,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
