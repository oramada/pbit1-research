"""Independent all-node probe and operator path replay for calibrated Transformer."""
import json,hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/calibrated_transformer')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Original CUDA backend required')
    p=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['native_operator_runs']==180
    for n,h in manifest['execution']['source_sha256'].items():assert sha(n)==h
    probes=operators=0;maxerr=0.;totalinfer=totalcal=0;errors=[]
    for stage in range(12):
        c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;size=int(np.prod(shape))
        low=np.repeat(a['lower'],128,0)[:shape[0]];span=np.repeat(a['span'],128,0)[:shape[0]];bias=a['bias'];nominal=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32)
        layer=PublicLayer(np.full(shape,.5));assert layer.n_free==size and not layer.clamped
        probe=layer.frequencies(2048);infer=layer.frequencies(1024)
        for seed in range(5):
            z=np.random.default_rng(np.random.SeedSequence([2026114600,stage,seed])).standard_normal((2,size));gain=np.exp(.2*z[1]-.02);offset=.2*z[0]
            cp=OUT/'calibration'/f'stage{stage}_seed{seed}.npz';cal=np.load(cp);cm=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==cm['archive_sha256'] and cm['controls_sha256']==sha(c['path'])
            assert cm['gain_sha256']==digest(gain) and cm['offset_sha256']==digest(offset);u=[]
            pf=(-4+np.array([96,159])*8/255).astype(np.float32).astype(float)
            for pi,field in enumerate(pf):
                fields=(gain*field+offset).astype(np.float32);assert digest(fields)==cm['probe_field_sha256'][pi]
                key=jax.random.key(1140731)
                for v in [seed,stage,pi]:key=jax.random.fold_in(key,v)
                q=np.asarray(probe(key[None],jnp.asarray(fields)))[0].reshape(-1);count=np.rint(q*2048).astype(np.uint16)
                np.testing.assert_array_equal(cal[f'count{pi}'],count);assert count.shape==(size,) and np.all(count<=2048)
                u.append(.5*(np.log(count.astype(float)+.5)-np.log(2048-count.astype(float)+.5)));probes+=1
            raw=(u[1]-u[0])/(pf[1]-pf[0]);ah=np.minimum(4,np.maximum(.25,raw));dh=(u[1]+u[0])/2-raw*(pf[1]+pf[0])/2
            requested=(nominal.astype(float)-dh)/ah;corrected=np.rint((np.clip(requested,-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(cal['corrected_codes'],corrected)
            assert cm['gain_guard_low']==np.count_nonzero(raw<.25) and cm['gain_guard_high']==np.count_nonzero(raw>4) and cm['commands_bound_clipped']==np.count_nonzero(np.abs(requested)>4)
            assert cm['free_nodes']==size and cm['clamped_nodes']==0 and cm['probe_runs']==2 and cm['free_evaluations']==4096*size;totalcal+=4096*size
            corrected=(-4+corrected.astype(float)*8/255).astype(np.float32)
            key=jax.random.key(1140730)
            keys=[]
            for bank in range(8):
                k=key
                for v in [seed,stage,bank]:k=jax.random.fold_in(k,v)
                keys.append(k)
            for condition in ['ideal','uncalibrated','cal4096']:
                folder=OUT/'runs'/f'{condition}_seed{seed}';mp=folder/f'stage{stage}_native_output.json';meta=json.loads(mp.read_text());sig=meta['signature'];out=folder/f'stage{stage}_native_output.npy';xp=folder/f'stage{stage}_inputs.npy';rp=folder/f'stage{stage}_conditional_reference.npy'
                assert sha(out)==meta['output_sha256'] and sha(xp)==sig['input_sha256'] and sha(rp)==sig['reference_sha256'] and sig['controls_sha256']==sha(c['path'])
                command=corrected if condition=='cal4096' else nominal;fields=command if condition=='ideal' else (gain*command.astype(float)+offset).astype(np.float32)
                assert digest(fields)==meta['field_sha256'] and meta['calibration_archive_sha256']==sha(cp)
                x,y,reference=np.load(xp),np.load(out),np.load(rp);assert x.shape==(64,65,shape[0]) and y.shape==reference.shape==(64,65,shape[1]) and np.isfinite(y).all()
                qq=np.asarray(infer(jnp.stack(keys),jnp.asarray(fields))).astype(float)
                xb=x.reshape(8,520,shape[0]).astype(float);expected=(xb@(low+span*qq)+bias).reshape(y.shape)
                err=float(np.max(np.abs(y.astype(float)-expected))/(1+np.max(np.abs(expected))));assert err<2e-5,(stage,seed,condition,err);maxerr=max(maxerr,err)
                actualq=(1/(1+np.exp(-2*fields.astype(float)))).reshape(shape);mean=x.astype(float)@(low+span*actualq)+bias
                energy=np.mean(x.astype(float)**2,axis=(0,1));pv=float(np.mean(energy@(span.astype(float)**2*actualq*(1-actualq)))/1024)
                values={'conditional_linear_mse':np.mean((y.astype(float)-reference)**2),'conditional_control_bias_mse':np.mean((mean-reference)**2),'predicted_sampling_mse':pv,'sampling_mse_vs_programmed_expectation':np.mean((y.astype(float)-mean)**2)}
                for name,value in values.items():np.testing.assert_allclose(meta[name],value,atol=1e-12,rtol=1e-11)
                assert meta['free_nodes']==size and meta['clamped_nodes']==0 and meta['bank_batches']==8 and meta['free_node_updates']==meta['logical_weight_readouts']==8*1024*size
                assert meta['binary_product_terms']==64*65*1024*size and meta['actual_nominal_command_bytes']==size and meta['metadata_range_bytes']==a['lower'].nbytes+a['span'].nbytes
                totalinfer+=8*1024*size;operators+=1;errors.append({'stage':stage,'seed':seed,'condition':condition,'max_scaled_output_error':err})
                print(f'Independent native replay stage{stage} seed{seed} {condition}, error={err:.3g}',flush=True)
        jax.clear_caches()
    assert probes==120 and operators==180 and totalinfer==manifest['total_native_inference_free_evaluations'] and totalcal==manifest['total_native_calibration_free_evaluations']
    audit={'passed':True,'native_operator_runs':operators,'native_probe_windows':probes,'every_native_weight_node_replayed':True,'all_field_control_fit_mse_and_resource_counts_checked':True,'max_scaled_output_error':maxerr,'tolerance':2e-5,'source_sha256':sha(__file__),'errors':errors,
        'scope':'Every calibrated Transformer operator and probe replayed with independently generated fields/keys and separately constructed public THRML frequency programs. Full-model task integration requires separate hook/score audit. Synthetic static conditions; no physical cost claim.'}
    (OUT/'native_path_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='errors'},indent=2))


if __name__=='__main__':main()
