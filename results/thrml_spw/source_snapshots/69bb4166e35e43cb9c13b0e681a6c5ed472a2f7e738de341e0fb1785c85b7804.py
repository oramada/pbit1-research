"""Public-factor count/operator replay using independently regenerated NN inputs."""
import json,sys,traceback,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha
ROOT=Path('results/thrml_spw/probe_placement')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Worker:
    def __init__(self):
        self.p=json.loads((ROOT/'protocol.json').read_text());self.designs=json.loads((ROOT/'placement_designs.json').read_text());self.layers={};self.fn={};self.errors=[];self.probes=0;self.calupdates=0;self.infupdates=0
        assert sha(ROOT/'placement_designs.json')==self.p['placement_designs_sha256']

    def execute(self,r):
        method,stage,seed=r['method'],int(r['stage']),int(r['seed']);c=self.p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;n=int(np.prod(shape))
        lo=np.repeat(a['lower'],128,0);sp=np.repeat(a['span'],128,0);bias=a['bias'];b=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32)
        if stage not in self.layers:
            self.layers[stage]=PublicLayer(np.full(shape,.5));self.fn[stage]={t:self.layers[stage].frequencies(t) for t in [2048,1024]}
            assert not self.layers[stage].clamped and self.layers[stage].n_free==n
        z=np.random.default_rng(np.random.SeedSequence([2026116600,stage,seed])).standard_normal((2,n));gain=np.exp(.2*z[1]-.02);offset=.2*z[0]
        pair=[96,159] if method=='fixed' else [self.designs[stage]['chosen'][k] for k in ['low_code','high_code']];pf=(-4+np.array(pair)*8/255).astype(np.float32).astype(float)
        cp=ROOT/'calibration'/f'{method}_stage{stage}_seed{seed}.npz';cal=np.load(cp);cm=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==cm['archive_sha256'] and cm['controls_sha256']==c['sha256'] and cm['probe_codes']==pair and cm['gain_sha256']==digest(gain) and cm['offset_sha256']==digest(offset)
        u=[]
        for pi,field in enumerate(pf):
            fields=(gain*field+offset).astype(np.float32);assert digest(fields)==cm['probe_field_sha256'][pi]
            key=jax.random.key(1160731)
            for v in [seed,stage,pi]:key=jax.random.fold_in(key,v)
            q=np.asarray(self.fn[stage][2048](key[None],jnp.asarray(fields)))[0].reshape(-1);count=np.rint(q*2048).astype(np.uint16)
            np.testing.assert_array_equal(cal[f'count{pi}'],count);u.append(.5*(np.log(count.astype(float)+.5)-np.log(2048-count.astype(float)+.5)));self.probes+=1
        raw=(u[1]-u[0])/(pf[1]-pf[0]);ah=np.clip(raw,.25,4);dh=(u[1]+u[0])/2-raw*(pf[1]+pf[0])/2
        requested=(b.astype(float)-dh)/ah;codes=np.rint((np.clip(requested,-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(cal['corrected_codes'],codes)
        cmd=(-4+codes.astype(float)*8/255).astype(np.float32);fields=(gain*cmd.astype(float)+offset).astype(np.float32);prob=1/(1+np.exp(-2*fields.astype(float)));p0=1/(1+np.exp(-2*b.astype(float)))
        for k,v in {'gain_RMSE':np.sqrt(np.mean((ah-gain)**2)),'offset_RMSE':np.sqrt(np.mean((dh-offset)**2)),'realized_weight_MSE_vs_nominal_program':np.mean(sp.reshape(-1).astype(float)**2*(prob-p0)**2),'realized_field_MSE_vs_nominal':np.mean((fields.astype(float)-b.astype(float))**2)}.items():np.testing.assert_allclose(cm[k],v,atol=1e-12,rtol=1e-11)
        assert cm['gain_guard_low']==np.count_nonzero(raw<.25) and cm['gain_guard_high']==np.count_nonzero(raw>4) and cm['requested_commands_clipped']==np.count_nonzero(np.abs(requested)>4) and cm['free_nodes']==n and cm['clamped_nodes']==0 and cm['free_node_evaluations']==4096*n
        folder=ROOT/'runs'/f'{method}_seed{seed}';yp=folder/f'stage{stage}_native_output.npy';meta=json.loads(yp.with_suffix('.json').read_text());man=json.loads(yp.with_suffix('.manifest.json').read_text());assert man['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and man['packages']['thrml']=='0.1.4'
        xp=Path(r['regenerated_inputs']);assert sha(xp)==meta['signature']['inputs_sha256']==r['expected_input_sha256'] and sha(yp)==meta['output_sha256'] and meta['signature']['calibration_archive_sha256']==sha(cp) and meta['field_sha256']==digest(fields) and meta['command_sha256']==digest(codes)
        x,y=np.load(xp),np.load(yp);assert x.shape==(64,65,shape[0]) and y.shape==(64,65,shape[1]) and x.dtype==y.dtype==np.float32
        keys=[]
        for bank in range(8):
            key=jax.random.key(1160730)
            for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
            keys.append(key)
        q=np.asarray(self.fn[stage][1024](jnp.stack(keys),jnp.asarray(fields))).astype(float);expected=(x.reshape(8,520,shape[0]).astype(float)@(lo+sp*q)+bias).reshape(y.shape)
        error=float(np.max(np.abs(y.astype(float)-expected))/(1+np.max(np.abs(expected))));assert error<2e-5,(method,seed,stage,error)
        pp=prob.reshape(shape);mean=x.astype(float)@(lo+sp*pp)+bias;pred=float(np.mean(np.mean(x.astype(float)**2,axis=(0,1))@(sp.astype(float)**2*pp*(1-pp)))/1024)
        np.testing.assert_allclose(meta['predicted_sampling_mse'],pred,atol=1e-12,rtol=1e-11);np.testing.assert_allclose(meta['sampling_mse_vs_programmed_expectation'],np.mean((y.astype(float)-mean)**2),atol=1e-12,rtol=1e-11)
        assert meta['free_node_updates']==meta['logical_weight_readouts']==8*1024*n and meta['binary_product_terms']==64*65*1024*n and meta['free_nodes']==n and meta['clamped_nodes']==0
        self.calupdates+=4096*n;self.infupdates+=8*1024*n;self.errors.append({'method':method,'seed':seed,'stage':stage,'scaled_output_error':error})
        return {'status':'ok','scaled_output_error':error}

    def finish(self):
        assert len(self.errors)==120 and self.probes==240 and self.calupdates==10*4096*21233664 and self.infupdates==10*8*1024*21233664
        r={'passed':True,'native_operator_runs':120,'native_probe_windows':240,'every_native_node_replayed':True,'all_regenerated_input_hashes_exact':True,'all_counts_fits_codes_diagnostics_and_resources_checked':True,'max_scaled_output_error':max(r['scaled_output_error'] for r in self.errors),'tolerance':2e-5,'calibration_free_updates_checked':self.calupdates,'inference_free_updates_checked':self.infupdates,'source_sha256':sha(__file__),'errors':self.errors,
           'scope':'Actual public THRML count/frequency replay, inputs independently regenerated and SHA256-matched to runtime data. No stored input/teacher duplication or replacement sampler. Model integration checked by separate pre/post-hook process.'}
        (ROOT/'native_path_audit.json').write_text(json.dumps(r,indent=2)+'\n');return {'status':'ok','operators':120,'probes':240}


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA required')
    worker=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            response=worker.finish() if r.get('action')=='finish' else worker.execute(r)
            print(json.dumps(response),flush=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)


if __name__=='__main__':main()
