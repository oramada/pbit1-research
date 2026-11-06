"""Independent public-factor frequency replay of all clocked probes/examples."""
import csv,hashlib,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/clocked_mnist');DATA=Path('data/thrml_spw/positive_mnist')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def hashes(a):return [digest(f) for f in a.reshape(-1,a.shape[-1])]


def public_dynamic(layer):
    fn=layer.frequencies(64)
    return jax.jit(jax.vmap(lambda k,f:fn(k[None],f)[0]))


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA original backend required')
    p=json.loads((OUT/'protocol.json').read_text());result=json.loads((OUT/'REPORT.json').read_text());assert result['native_full_test_runs']==150
    for name,h in p['source_sha256'].items():assert sha(name)==h
    d=np.load(DATA/'evaluation_data.npz');ids=np.random.default_rng(2026110301).permutation(10000);x=d['test_inputs'][ids];xb=np.pad(x,((0,16),(0,0))).reshape(313,32,64)
    # Independent arithmetic specifies every occupied clock interval.
    batches=np.arange(313);i0=batches[:,None]*32+(batches[:,None]//32+1)*64+np.arange(16)[None,:];i1=i0+16
    cal_indices=np.arange(10)[:,None]*1088+np.arange(64)[None,:]
    occupied=np.concatenate([cal_indices.reshape(-1),i0.reshape(-1),i1.reshape(-1)])
    np.testing.assert_array_equal(np.sort(occupied),np.arange(10656))
    errors=[];cal_audited=0;field_hashes=0
    for model in range(5):
        ex=np.load(DATA/f'seed{model}.npz');bias=[ex[f'bias_{i}'] for i in range(2)]
        # Public native state programs have exactly one free node per weight,
        # independent of target probability encoding and production classes.
        layers=[PublicLayer(np.full((64,32),.5)),PublicLayer(np.full((32,10),.5))]
        for layer in layers:assert not layer.clamped and layer.n_free==layer.size
        funcs=[public_dynamic(layer) for layer in layers];probe_fn=public_dynamic(PublicLayer(np.full(2368,.5)))
        for device in range(5):
            noise=np.random.default_rng(2026110400+100*model+device).standard_normal((2,2368));gain=np.exp(.2*noise[1]-.02);offset=.2*noise[0]
            increments=np.random.default_rng(2026110500+100*model+device).standard_normal((10656,2368))
            path=np.concatenate([np.zeros((1,2368)),np.cumsum(increments,axis=0)],axis=0)/4;del increments
            infer_key=jax.random.key(1100730)
            for v in [model,device]:infer_key=jax.random.fold_in(infer_key,v)
            kk=[]
            for layer in [0,1]:
                kk.append(jax.vmap(lambda b:jax.vmap(lambda s:jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(infer_key,b),layer),s))(jnp.arange(16)))(jnp.arange(313)).reshape(-1))
            for rate in [0.,.01,.04]:
                tag=f'model{model}_device{device}_rate{rate:g}';cp=OUT/'calibration'/f'{tag}.npz';bundle=np.load(cp);meta=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==meta['archive_sha256'] and digest(path)==meta['path_sha256']
                probe_codes=np.rint((np.array([-1.,1.])+4)*255/8);probe_fields=(-4+probe_codes*8/255).astype(np.float32)
                commands=np.repeat(probe_fields,32)[None,:,None]
                fields=(gain*commands+offset+rate*path[cal_indices]).astype(np.float32)
                assert [hashes(f) for f in fields]==meta['calibration_field_hashes'];field_hashes+=640
                ckeys=[]
                for window in range(10):
                    key=jax.random.key(1100731)
                    for v in [model,device,window]:key=jax.random.fold_in(key,v)
                    ckeys.append(jax.vmap(lambda s:jax.random.fold_in(key,s))(jnp.arange(64)))
                q=np.asarray(probe_fn(jnp.stack(ckeys).reshape(-1),jnp.asarray(fields.reshape(-1,2368))))
                counts=np.rint(q*64).astype(np.uint8).reshape(10,64,2368)
                np.testing.assert_array_equal(counts,bundle['segment_counts']);cal_audited+=10;del q,fields
                for policy in ['retain_initial','refresh_every32_batches']:
                    name=f'{tag}_{policy}';rn=OUT/'runs'/f'{name}.npz';raw=np.load(rn);row=json.loads(rn.with_suffix('.json').read_text());assert sha(rn)==row['archive_sha256']
                    index=np.zeros(313,dtype=int) if policy=='retain_initial' else np.arange(313)//32
                    commands=(-4+bundle['corrected_codes'][index].astype(float)*8/255).astype(np.float32)
                    f0=(gain[:2048]*commands[:,None,:2048]+offset[:2048]+rate*path[i0,:2048]).astype(np.float32)
                    f1=(gain[2048:]*commands[:,None,2048:]+offset[2048:]+rate*path[i1,2048:]).astype(np.float32)
                    assert [hashes(f0),hashes(f1)]==row['layer_field_hashes'];field_hashes+=10016
                    q0=np.asarray(funcs[0](kk[0],jnp.asarray(f0.reshape(-1,2048)))).reshape(313,16,64,32).mean(axis=1)
                    expected=(np.einsum('bni,bio->bno',xb.astype(float),q0.astype(float))+bias[0]).reshape(-1,32)[:10000]
                    pe=float(np.max(np.abs(expected-raw['preactivation0']))/(1+np.max(np.abs(expected))));assert pe<2e-5,(name,'pre',pe)
                    hidden=np.pad(np.maximum(raw['preactivation0'],0),((0,16),(0,0))).reshape(313,32,32)
                    q1=np.asarray(funcs[1](kk[1],jnp.asarray(f1.reshape(-1,320)))).reshape(313,16,32,10).mean(axis=1)
                    expected=(np.einsum('bni,bio->bno',hidden.astype(float),q1.astype(float))+bias[1]).reshape(-1,10)[:10000]
                    le=float(np.max(np.abs(expected-raw['logits']))/(1+np.max(np.abs(expected))));assert le<2e-5,(name,'logits',le)
                    errors.append({'model':model,'device':device,'rate':rate,'policy':policy,'scaled_preactivation_error':pe,'scaled_logit_error':le})
                    del f0,f1,q0,q1
                print(f'Public full-clock probes/NN replay {tag}',flush=True)
            del path
        jax.clear_caches()
    assert len(errors)==150 and cal_audited==750 and field_hashes==1550400
    with (OUT/'public_path_errors.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(errors[0]));wr.writeheader();wr.writerows(errors)
    audit={'passed':True,'native_full_test_runs':150,'examples_per_run':10000,'calibration_events':750,'probe_windows':1500,'native_field_hashes_regenerated':field_hashes,
        'max_scaled_preactivation_error':max(r['scaled_preactivation_error'] for r in errors),'max_scaled_logit_error':max(r['scaled_logit_error'] for r in errors),'tolerance':2e-5,
        'source_sha256':sha(__file__),'public_layer_source_sha256':sha('research/thrml_spw/calibrated_mnist_path_audit.py'),
        'scope':'Independent serial clock/path/field/key generation and separately constructed public THRML all-free programs. Every stored probe segment count and every example in both NN layers replayed. Same CUDA backend; synthetic drift, no physical or stationary-policy proof.'}
    (OUT/'public_path_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
