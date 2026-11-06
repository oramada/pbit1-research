"""Every known-parameter oracle operator replayed through public THRML factors."""
from pathlib import Path
import json,hashlib
import numpy as np
import jax
import jax.numpy as jnp
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/transformer_calibration_floor')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Original CUDA backend required')
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['native_operator_runs']==60 and m['native_probe_runs']==0
    for n,h in p['source_sha256'].items():assert sha(n)==h
    errors=[];updates=0
    for stage,c in enumerate(p['control_artifacts']):
        assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;size=int(np.prod(shape))
        b=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);lo=np.repeat(a['lower'],128,0);span=np.repeat(a['span'],128,0);bias=a['bias']
        public=PublicLayer(np.full(shape,.5));assert public.n_free==size and not public.clamped;fn=public.frequencies(1024)
        for seed in range(5):
            noise=np.random.default_rng(np.random.SeedSequence([2026114600,stage,seed])).standard_normal((2,size));gain=np.exp(.2*noise[1]-.02);offset=.2*noise[0]
            inverse=(b.astype(float)-offset)/gain;codes=np.rint((np.clip(inverse,-4,4)+4)*255/8).astype(np.uint8);command=(-4+codes.astype(float)*8/255).astype(np.float32);fields=(gain*command.astype(float)+offset).astype(np.float32)
            folder=OUT/'runs'/f'oracle8_seed{seed}';yp=folder/f'stage{stage}_native_output.npy';xp=folder/f'stage{stage}_inputs.npy';meta=json.loads(yp.with_suffix('.json').read_text());manifest=json.loads(yp.with_suffix('.manifest.json').read_text())
            assert manifest['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and manifest['packages']['thrml']=='0.1.4'
            assert sha(yp)==meta['output_sha256'] and sha(xp)==meta['signature']['inputs_sha256'] and sha(c['path'])==meta['signature']['controls_sha256']
            assert meta['signature']['condition']=='oracle8' and meta['field_sha256']==digest(fields) and meta['oracle_command_sha256']==digest(codes)
            keys=[]
            for bank in range(8):
                key=jax.random.key(1140730)
                for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
                keys.append(key)
            q=np.asarray(fn(jnp.stack(keys),jnp.asarray(fields))).astype(float)
            x,y=np.load(xp),np.load(yp);assert x.shape==(64,65,shape[0]) and y.shape==(64,65,shape[1]) and x.dtype==y.dtype==np.float32
            xb=x.reshape(8,520,shape[0]).astype(float);expected=(xb@(lo+span*q)+bias).reshape(y.shape)
            err=float(np.max(np.abs(y.astype(float)-expected))/(1+np.max(np.abs(expected))));assert err<2e-5,(stage,seed,err)
            qp=(1/(1+np.exp(-2*fields.astype(float)))).reshape(shape);qn=(1/(1+np.exp(-2*b.astype(float)))).reshape(shape)
            mean=x.astype(float)@(lo+span*qp)+bias;nominal=x.astype(float)@(lo+span*qn)+bias
            values={'mean_bias_mse_vs_nominal_program':float(np.mean((mean-nominal)**2)),'sampling_mse_vs_programmed_expectation':float(np.mean((y.astype(float)-mean)**2)),'predicted_sampling_mse':float(np.mean(np.mean(x.astype(float)**2,axis=(0,1))@(span.astype(float)**2*qp*(1-qp)))/1024)}
            for k,v in values.items():np.testing.assert_allclose(meta[k],v,atol=1e-12,rtol=1e-11)
            assert meta['known_inverse_bound_clipped']==np.count_nonzero(np.abs(inverse)>4) and meta['free_nodes']==size and meta['clamped_nodes']==0
            assert meta['bank_batches']==8 and meta['free_node_updates']==meta['logical_weight_readouts']==8*1024*size and meta['binary_product_terms']==64*65*1024*size
            errors.append({'stage':stage,'seed':seed,'scaled_output_error':err});updates+=8*1024*size
            print(f'Public known-parameter oracle replay stage{stage} seed{seed}: error={err:.3g}',flush=True)
        jax.clear_caches()
    assert len(errors)==60 and updates==m['native_free_evaluations']
    result={'passed':True,'native_operator_runs':60,'native_probe_runs':0,'every_weight_node_independently_replayed':True,'max_scaled_output_error':max(r['scaled_output_error'] for r in errors),'tolerance':2e-5,'native_free_evaluations_checked':updates,'source_sha256':sha(__file__),'errors':errors,
        'scope':'Known-parameter diagnostic only, finite8bit[-4,4] control andactualTHRML T1024 samples onoldconsumedtext/profiles/keys. No measured-calibration algorithm or fresh confirmation claim. Model integration independently checked by hooks.'}
    (OUT/'native_path_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2))


if __name__=='__main__':main()
