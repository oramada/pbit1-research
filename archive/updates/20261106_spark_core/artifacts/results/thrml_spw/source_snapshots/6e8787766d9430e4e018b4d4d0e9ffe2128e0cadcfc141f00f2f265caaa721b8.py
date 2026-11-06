"""Frozen all-matrix native operator comparison, immutable per-case outputs."""
import os,json,time
import numpy as np
import jax
import jax.numpy as jnp
from .positional_operator_common import OUT,sha,digest,guard,controls,prediction
from .positional_operator_native import PositionalOperator
from .block_cached_control_worker import atomic_json,atomic_npy
from .provenance import manifest

def main():
    assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0' and jax.default_backend()=='gpu' and not jax.config.x64_enabled
    p=json.loads((OUT/'protocol.json').read_text());guard(p);assert not (OUT/'manifest.json').exists();banks={};rows=[];start=time.monotonic()
    for entry in p['matrices']:
        stage=entry['stage'];anchor=entry['anchor'];assert sha(anchor['inputs'])==anchor['inputs_sha256'] and sha(anchor['target'])==anchor['target_sha256'];x=np.load(anchor['inputs']);target=np.load(anchor['target']);assert x.shape==(64,entry['shape'][0]);xx=jnp.asarray(x)
        for k,which,budgets in [(1,'single_controls',p['single_samples']),(4,'positional_controls',p['positional_samples'])]:
            c=entry[which];assert sha(c['path'])==c['sha256'];shape,lo,sp,bias,fields=controls(c,k);signature={'protocol_sha256':sha(OUT/'protocol.json'),'controls_sha256':c['sha256'],'inputs_sha256':anchor['inputs_sha256'],'target_sha256':anchor['target_sha256'],'fields_sha256':digest(fields),'stage':stage,'K':k}
            cache=(shape,k)
            if cache not in banks:banks[cache]=PositionalOperator(shape,k)
            layer=banks[cache];args=[jnp.asarray(z) for z in [fields.ravel(),lo,sp,bias]]
            for t in budgets:
                mean,predicted=prediction(x,target,lo,sp,bias,fields,k,t)
                for seed in p['seeds']:
                    name=f'stage{stage}_K{k}_T{t}_seed{seed}';path=OUT/'outputs'/f'{name}.npy';meta=path.with_suffix('.json');sig={**signature,'T':t,'seed':seed}
                    if meta.exists():
                        row=json.loads(meta.read_text());assert row['signature']==sig and sha(path)==row['output_sha256']
                    else:
                        key=jax.random.key(p['key_root'])
                        for v in [stage,k,seed]:key=jax.random.fold_in(key,v)
                        y=np.asarray(layer.forward(t)(key,xx,*args));assert y.shape==target.shape and y.dtype==np.float32 and np.isfinite(y).all();atomic_npy(path,y);n=entry['weights'];row={'case':name,'signature':sig,'output':str(path),'output_sha256':sha(path),'free_nodes':k*n,'clamped_nodes':0,'sample_rounds':t,'native_node_updates':k*n*t,'binary_input_terms':64*k*n*t,'converted_control_range_bias_payload_bits':n*(8 if k==1 else 4)+64*(n//16)+32*len(bias),'observed_target_MSE':float(np.mean((y.astype(float)-target)**2)),'observed_MSE_vs_programmed_mean':float(np.mean((y.astype(float)-mean)**2)),**predicted};atomic_json(meta,row)
                    rows.append(row)
                print(f'Operator stage{stage} K{k} T{t} seeds0..4 complete elapsed={time.monotonic()-start:.1f}s',flush=True)
        guard(p)
    assert len(rows)==p['native_operators'];r={'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'native_operators':len(rows),'native_node_updates':sum(v['native_node_updates'] for v in rows),'all_free_no_clamps':True,'full_neural_models':0,'rows':rows,'elapsed_seconds':time.monotonic()-start,'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'scope':p['scope']};atomic_json(OUT/'statistics.json',r);manifest(OUT/'manifest.json',p,{k:v for k,v in r.items() if k!='rows'});print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
