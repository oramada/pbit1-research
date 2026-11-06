"""Every-example public-factor replay for clamped and entirely free encodings."""
import csv,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/finite_endpoint_mnist');DATA=Path('data/thrml_spw/positive_mnist')


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Original CUDA backend required')
    p=json.loads((OUT/'protocol.json').read_text())
    for path,h in p['source_sha256'].items():assert sha(path)==h
    assert sha(DATA/'evaluation_data.npz')==p['input_sha256']
    x=np.load(DATA/'evaluation_data.npz')['test_inputs'];xb=np.pad(x,((0,16),(0,0))).reshape(313,32,64)
    with (OUT/'native_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==1600;errors=[]
    for model in range(5):
        ex=np.load(DATA/f'seed{model}.npz');original=[ex[f'weight_{i}'] for i in range(2)];bias=[ex[f'bias_{i}'] for i in range(2)]
        for encoding in p['encodings']:
            en=encoding['name'];ep=1/(1+np.exp(2*encoding['bound']))
            weights=[a.astype(float) if encoding['clamp_endpoints'] else np.minimum(1-ep,np.maximum(ep,a.astype(float))) for a in original]
            layers=[PublicLayer(a) for a in weights];split=layers[0].n_free
            if not encoding['clamp_endpoints']:assert sum(l.n_free for l in layers)==2368 and all(not l.clamped for l in layers)
            fn={(i,t):layers[i].frequencies(t) for i in [0,1] for t in [256,1024]}
            for device in range(10):
                b=np.load(OUT/'devices'/f'model{model}_device{device}_{en}.npz')
                for t in [256,1024]:
                    key=jax.random.key(1090730)
                    for v in [model,device,t]:key=jax.random.fold_in(key,v)
                    keys=jax.vmap(lambda i:jax.random.fold_in(key,i))(jnp.arange(313));kk=jax.vmap(lambda k:jax.random.split(k,2))(keys)
                    for condition in p['conditions']:
                        fields=b[f'fields_{condition}'];name=f'model{model}_device{device}_{en}_{condition}_T{t}'
                        raw=np.load(OUT/'runs'/f'{name}.npz');pre,logits=raw['preactivation0'],raw['logits']
                        q0=np.asarray(fn[0,t](kk[:,0],jnp.asarray(fields[:split])))
                        expected=(np.einsum('bni,bio->bno',xb.astype(float),q0.astype(float))+bias[0]).reshape(-1,32)[:10000]
                        e0=float(np.max(np.abs(expected-pre))/(1+np.max(np.abs(expected))));assert e0<2e-5,(name,'layer0',e0)
                        hb=np.pad(np.maximum(pre,0),((0,16),(0,0))).reshape(313,32,32)
                        q1=np.asarray(fn[1,t](kk[:,1],jnp.asarray(fields[split:])))
                        expected=(np.einsum('bni,bio->bno',hb.astype(float),q1.astype(float))+bias[1]).reshape(-1,10)[:10000]
                        e1=float(np.max(np.abs(expected-logits))/(1+np.max(np.abs(expected))));assert e1<2e-5,(name,'layer1',e1)
                        errors.append({'model':model,'device':device,'encoding':en,'condition':condition,'T':t,'scaled_preactivation_error':e0,'scaled_logit_error':e1})
                if device%5==4:print(f'Public all-example replay model{model} {en} through device{device}',flush=True)
            jax.clear_caches()
    assert len(errors)==1600
    with (OUT/'public_path_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
    audit={'passed':True,'native_runs_audited':1600,'examples_per_run':10000,'layers_per_run':2,
        'max_scaled_preactivation_error':max(r['scaled_preactivation_error'] for r in errors),
        'max_scaled_logit_error':max(r['scaled_logit_error'] for r in errors),'tolerance':2e-5,
        'source_sha256':sha(__file__),'public_layer_source_sha256':sha('research/thrml_spw/calibrated_mnist_path_audit.py'),
        'scope':'Every example in every case reconstructed from independently specified finite-endpoint weights and public THRML integer state frequencies. Entirely free variants independently assert zero clamp blocks. Same CUDA backend, no physical claim.'}
    (OUT/'public_path_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
