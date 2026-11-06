"""Independent public THRML replay on regenerated complete student activations."""
import json,sys,traceback,os,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .fine_range_common import OUT,condition
from .calibrated_mnist_path_audit import PublicLayer
from .range_fixed_replay_cohort import public_function
from .block_cached_control_worker import sha,atomic_json
from .packed_float_outputs import load_output
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
class Auditor:
    def __init__(self):
        assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';self.p=json.loads((OUT/'protocol.json').read_text());self.execution=json.loads((OUT/'execution.json').read_text());self.layers={};self.fn={};self.freq={};self.rows=[]
    def execute(self,r):
        for name,h in json.loads((OUT/'audit_execution.json').read_text())['source_sha256'].items():assert sha(name)==h
        yp,xp=Path(r['output']),Path(r['inputs']);meta=json.loads(yp.with_suffix('.json').read_text());sig=meta['signature'];stage,seed=int(sig['stage']),int(sig['seed']);cfg=condition(self.p,sig['condition']);g,t=cfg['group'],cfg['samples'];assert sig['samples']==t and sig['group']==g and stage in range(24) and seed in range(5) and sig['XLA_FLAGS']==os.environ['XLA_FLAGS'] and sig['execution_sha256']==sha(OUT/'execution.json');c=self.p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256']==sig['controls_sha256'] and sha(xp)==r['input_sha256']==sig['inputs_sha256'] and sha(yp)==meta['output_sha256']
        for name,h in self.execution['source_sha256'].items():assert sha(name)==h
        manifest=json.loads(yp.with_suffix('.manifest.json').read_text());assert manifest['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and manifest['packages']['thrml']=='0.1.4'
        a=np.load(c['path']);shape=a['codes'].shape;n=int(np.prod(shape));lo=np.repeat(a['lower'],g,0);span=np.repeat(a['span'],g,0);fields=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);assert digest(fields)==meta['field_sha256'];x,y=np.load(xp),load_output(yp);assert x.shape==(32,65,shape[0]) and y.shape==(32,65,shape[1]) and x.dtype==y.dtype==np.float32
        if shape not in self.layers:self.layers[shape]=PublicLayer(np.full(shape,.5))
        layer=self.layers[shape];assert layer.n_free==n and not layer.clamped
        if (stage,g,t) not in self.fn:self.fn[stage,g,t]=public_function(layer,jnp.asarray(span),jnp.asarray(lo),jnp.asarray(a['bias']),shape,t);self.freq[stage,g,t]=layer.frequencies(t)
        outputs=[];counts=[]
        for bank in range(4):
            key=jax.random.key(self.p['key_root'])
            for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
            yy=self.fn[stage,g,t](key,jnp.asarray(x[bank*8:(bank+1)*8].reshape(520,shape[0])),jnp.asarray(fields));outputs.append(np.asarray(yy).reshape(8,65,-1));q=np.asarray(self.freq[stage,g,t](key[None],jnp.asarray(fields)))[0];counts.append(np.rint(q.astype(float)*t).astype(np.int32))
        reproduced=np.concatenate(outputs);np.testing.assert_array_equal(reproduced,y);counts=np.stack(counts);assert np.all((counts>=0)&(counts<=t));reference=(x.reshape(4,520,shape[0]).astype(float)@(lo.astype(float)+span.astype(float)*counts.astype(float)/t)+a['bias']).reshape(y.shape);error=np.abs(y.astype(float)-reference);scaled=float(error.max()/(1+np.max(np.abs(reference))));u=2.**-24;d=shape[0];gd=2*d*u/(1-2*d*u);gt=(t-1)*u/(1-(t-1)*u);g2=2*u/(1-2*u);mag=np.abs(x.astype(float))@np.abs(span.astype(float));center=np.abs(x.astype(float))@np.abs(lo.astype(float));bias=np.abs(a['bias'].astype(float));rg=(2*d+4)*2.**-53/(1-(2*d+4)*2.**-53);bound=(gd+gt*(1+gd))*mag+gd*center+g2*((1+gt)*(1+gd)*mag+(1+gd)*center+bias)+(4*d+2*t+8)*np.finfo(np.float32).tiny+rg*(mag+center+bias);assert np.max(t*(mag+center+bias))<np.finfo(np.float32).max and np.all(error<=bound)
        prob=(1/(1+np.exp(-2*fields.astype(float)))).reshape(shape);mean=x.astype(float)@(lo.astype(float)+span.astype(float)*prob)+a['bias'];prediction=float(np.mean((x.astype(float)**2)@(span.astype(float)**2*prob*(1-prob)))/t);np.testing.assert_allclose(meta['predicted_sampling_mse'],prediction,rtol=1e-11,atol=1e-12);np.testing.assert_allclose(meta['sampling_mse_vs_programmed_expectation'],np.mean((y.astype(float)-mean)**2),rtol=1e-11,atol=1e-12);assert meta['free_nodes']==n and meta['clamped_nodes']==0 and meta['bank_batches']==4 and meta['free_node_updates']==meta['logical_weight_readouts']==4*t*n and meta['binary_product_terms']==2080*t*n
        row={'condition':sig['condition'],'stage':stage,'seed':seed,'group':g,'samples':t,'public_F32_output_bitwise_exact':True,'counts_sha256':digest(counts),'scaled_F64_frequency_error':scaled,'descriptive_2e_5_flag':bool(scaled<2e-5),'max_error_over_derived_bound':float(np.max(error/np.maximum(bound,1e-30))),'production_observed_updates_checked':4*t*n,'observed_updates_replayed':8*t*n};self.rows.append(row);return {'status':'ok','bitwise_exact':True}
    def finish(self):
        expected={(c,s,k) for c in ['coarse','fine'] for s in range(5) for k in range(24)};assert len(self.rows)==240 and {(r['condition'],r['seed'],r['stage']) for r in self.rows}==expected;r={'passed':True,'native_operators':240,'all_public_F32_outputs_bitwise_exact':True,'all_IEEE_envelopes_checked':True,'descriptive_2e_5_failures':sum(not r['descriptive_2e_5_flag'] for r in self.rows),'native_updates_replayed':sum(r['observed_updates_replayed'] for r in self.rows),'source_sha256':sha(__file__),'rows':self.rows};atomic_json(OUT/'native_audit.json',r);return {'status':'ok','operators':240}
def main():
    worker=Auditor();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(worker.finish() if r.get('action')=='finish' else worker.execute(r)),flush=True)
        except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
