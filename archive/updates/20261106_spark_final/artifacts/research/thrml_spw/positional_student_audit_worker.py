"""Public factors/sum-only and separate frequency audit of causal bank requests."""
import json,sys,traceback,os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .calibrated_mnist_path_audit import PublicLayer
from .verify_positional_operators import build
from .positional_student_common import OUT,sha,guard,condition
from .positional_operator_common import digest
from .block_cached_control_worker import atomic_json
class Auditor:
    def __init__(self):
        assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';self.p=json.loads((OUT/'protocol.json').read_text());guard(self.p);self.layers={};self.fn={};self.freq={};self.rows=[]
    def execute(self,r):
        guard(self.p);assert sha(r['input'])==r['input_sha256'];yp=Path(r['output']);meta=json.loads(yp.with_suffix('.json').read_text());sig=meta['signature'];assert sig['protocol_sha256']==sha(OUT/'protocol.json') and sig['input_sha256']==r['input_sha256'] and sha(yp)==meta['output_sha256'];cfg=condition(self.p,sig['condition']);k,t=cfg['K'],cfg['T'];assert sig['K']==k and sig['T']==t;stage,seed,bank=[sig[v] for v in ['stage','seed','bank']];entry=self.p['matrices'][stage];c=entry['single_controls' if k==1 else 'positional_controls'];assert sha(c['path'])==c['sha256']==sig['controls_sha256'];a=np.load(c['path']);shape=a['codes'].shape;lo=np.repeat(a['lower'],16,0);sp=np.repeat(a['span'],16,0);bias=a['bias'];fields=((-4+a['codes'].astype(float)*8/255).astype(np.float32)[...,None] if k==1 else np.where((a['codes'][...,None]//2**np.arange(4))%2,4,-4).astype(np.float32));assert digest(fields)==meta['fields_sha256'];x=np.load(r['input']);assert x.shape==(8,65,shape[0]);xx=x.reshape(520,-1)
        # Decode packed F32 bytes independently, without production load_output.
        with np.load(yp) as z:
            assert str(z['tag'])=='spw-f32-byteplanes-v1';dims=tuple(z['shape']);planes=z['byteplanes'];assert dims==(8,65,shape[1]) and planes.dtype==np.uint8 and planes.shape==(4,int(np.prod(dims)));flat=np.empty(int(np.prod(dims))*4,np.uint8)
            for j in range(4):flat[j::4]=planes[j]
            y=flat.view('<f4').reshape(dims)
        cache=(shape,k)
        if cache not in self.layers:self.layers[cache]=PublicLayer(np.full(shape+(k,),.5))
        layer=self.layers[cache];assert layer.n_free==int(np.prod(shape))*k and not layer.clamped
        if (cache,t) not in self.fn:self.fn[cache,t]=build(layer,shape,k,t);self.freq[cache,t]=layer.frequencies(t)
        key=jax.random.key(self.p['key_root'])
        for value in [seed,stage,k,bank]:key=jax.random.fold_in(key,value)
        jx,jf,jl,js,jb=[jnp.asarray(z) for z in [xx,fields.ravel(),lo,sp,bias]];replay=np.asarray(self.fn[cache,t](key,jx,jf,jl,js,jb)).reshape(dims);exact=bool(np.array_equal(replay,y));frequencies=np.asarray(self.freq[cache,t](key[None],jf))[0];counts=np.rint(frequencies.astype(float)*t).astype(np.int32);assert np.all((counts>=0)&(counts<=t));coeff=2.**np.arange(k)/((1<<k)-1);normalized=np.sum(counts*coeff,axis=-1)/t;reference=xx.astype(float)@(lo.astype(float)+sp.astype(float)*normalized)+bias;error=np.abs(y.reshape(reference.shape).astype(float)-reference);u=2.**-24;gamma=lambda n:n*u/(1-n*u);gd,gt,g4=gamma(2*shape[0]+4),gamma(t-1),gamma(4);mag=np.abs(xx.astype(float))@np.abs(sp.astype(float));center=np.abs(xx.astype(float))@np.abs(lo.astype(float));ab=np.abs(bias.astype(float));bound=(gd+gt*(1+gd))*mag+gd*center+g4*((1+gt)*(1+gd)*mag+(1+gd)*center+ab)+(4*shape[0]+2*t+16)*np.finfo(np.float32).tiny+((4*shape[0]+16)*2.**-53)*(mag+center+ab);ieee=bool(np.all(error<=bound));assert np.max(t*(mag+center+ab))<np.finfo(np.float32).max;expected=t*k*entry['weights'];assert meta['native_node_updates']==expected and meta['free_nodes']==k*entry['weights'] and meta['clamped_nodes']==0 and meta['binary_input_terms']==520*expected
        row={'condition':sig['condition'],'stage':stage,'seed':seed,'bank':bank,'public_F32_exact':exact,'frequency_IEEE_envelope_passed':ieee,'counts_sha256':digest(counts),'max_error_over_bound':float(np.max(error/np.maximum(bound,1e-30))),'independent_native_updates':2*expected};self.rows.append(row);return {'status':'ok','bitwise_exact':exact,'IEEE_envelope_passed':ieee}
    def finish(self):
        expected={(c['name'],s,stage,b) for c in self.p['conditions'] for s in self.p['sampling_seeds'] for stage in range(24) for b in range(4)};assert len(self.rows)==960 and {(r['condition'],r['seed'],r['stage'],r['bank']) for r in self.rows}==expected;r={'passed':all(v['public_F32_exact'] and v['frequency_IEEE_envelope_passed'] for v in self.rows),'bank_calls':960,'independent_native_updates':sum(v['independent_native_updates'] for v in self.rows),'protocol_sha256':sha(OUT/'protocol.json'),'production_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'rows':self.rows};assert r['independent_native_updates']==869730877440;atomic_json(OUT/'native_audit.json',r);return {'status':'ok','bank_calls':960,'passed':r['passed']}
def main():
    worker=Auditor();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action')=='quit':break
            if request.get('action')=='finish':result=worker.finish()
            else:result=worker.execute(request)
            print(json.dumps(result),flush=True)
        except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
