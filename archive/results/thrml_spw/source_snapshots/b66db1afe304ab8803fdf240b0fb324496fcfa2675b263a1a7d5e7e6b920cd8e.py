"""Independent full-matrix public THRML replay, frequency identity and roundoff."""
import json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,FactorSamplingProgram,SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .calibrated_mnist_path_audit import PublicLayer
from .block_cached_control_worker import sha,atomic_json
OUT=Path('results/thrml_spw/range_localization')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
class SumOnly(AbstractObserver):
    readout:object
    inputs:jax.Array
    span:jax.Array
    shape:tuple
    def init(self):return jnp.zeros((self.inputs.shape[0],self.shape[1]),jnp.float32)
    def __call__(self,program,free,clamped,carry,iteration):
        state=block_state_to_global(free+clamped,program.gibbs_spec);bits=from_global_state(state,program.gibbs_spec,[self.readout])[0]
        return carry+self.inputs@(bits.reshape(self.shape).astype(jnp.float32)*self.span),None

def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'native_manifest.json').read_text());assert m['native_operators']==360 and m['full_stochastic_neural_models_executed']==0;layers={};fn={};freq={};rows=[]
    for case in m['cases']:
        stage,seed=case['stage'],case['seed'];cond=next(c for c in p['native_conditions'] if c['name']==case['condition']);g,t=cond['group'],cond['samples'];c=p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;n=int(np.prod(shape));lo=np.repeat(a['lower'],g,0);sp=np.repeat(a['span'],g,0);b=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);yp=Path(case['output']);meta=json.loads(Path(case['metadata']).read_text());sig=meta['signature'];man=json.loads(yp.with_suffix('.manifest.json').read_text());assert man['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and man['packages']['thrml']=='0.1.4';assert sig['stage']==stage and sig['seed']==seed and sig['group']==g and sig['samples']==t and sig['condition']==case['condition'];assert sha(case['inputs'])==sig['inputs_sha256'] and sha(case['target'])==sig['target_sha256'] and sha(yp)==meta['output_sha256'] and sig['controls_sha256']==c['sha256'] and meta['field_sha256']==digest(b)
        for name,h in sig['source_sha256'].items():assert sha(name)==h
        x,y,target=np.load(case['inputs']),np.load(yp),np.load(case['target']);assert x.shape==(64,shape[0]) and y.shape==(64,shape[1]) and x.dtype==y.dtype==np.float32 and target.dtype==np.float64
        cache=tuple(shape)
        if cache not in layers:layers[cache]=PublicLayer(np.full(shape,.5));assert layers[cache].n_free==n and not layers[cache].clamped
        if (stage,g,t) not in fn:
            layer=layers[cache];span=jnp.asarray(sp);lower=jnp.asarray(lo);bias=jnp.asarray(a['bias'])
            def run(key,xx,fields):
                program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,fields)],[]);obs=SumOnly(layer.readout,xx,span,shape);total,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),layer.initial,[],obs.init(),obs);return total/t+xx@lower+bias
            fn[stage,g,t]=jax.jit(run);freq[stage,g,t]=layer.frequencies(t)
        key=jax.random.key(1200730)
        for v in [seed,stage]:key=jax.random.fold_in(key,v)
        replay=np.asarray(fn[stage,g,t](key,jnp.asarray(x),jnp.asarray(b)));np.testing.assert_array_equal(replay,y);q=np.asarray(freq[stage,g,t](key[None],jnp.asarray(b)))[0].astype(float);counts=np.rint(q*t).astype(np.int32);assert np.all((counts>=0)&(counts<=t));reference=x.astype(float)@(lo.astype(float)+sp.astype(float)*(counts.astype(float)/t))+a['bias'];error=np.abs(y.astype(float)-reference);scaled=float(error.max()/(1+np.max(np.abs(reference))));u=2.**-24;d=shape[0];gd=2*d*u/(1-2*d*u);gt=(t-1)*u/(1-(t-1)*u);g2=2*u/(1-2*u);mag=np.abs(x.astype(float))@np.abs(sp.astype(float));center=np.abs(x.astype(float))@np.abs(lo.astype(float));bias=np.abs(a['bias'].astype(float));refg=(2*d+4)*2.**-53/(1-(2*d+4)*2.**-53);bound=(gd+gt*(1+gd))*mag+gd*center+g2*((1+gt)*(1+gd)*mag+(1+gd)*center+bias)+(4*d+2*t+8)*np.finfo(np.float32).tiny+refg*(mag+center+bias);assert np.max(t*(mag+center+bias))<np.finfo(np.float32).max and np.all(error<=bound)
        pp=(1/(1+np.exp(-2*b.astype(float)))).reshape(shape);mean=x.astype(float)@(lo.astype(float)+sp.astype(float)*pp)+a['bias'];variance=float(np.mean((x.astype(float)**2)@(sp.astype(float)**2*pp*(1-pp)))/t);bias_mse=float(np.mean((mean-target)**2));values={'predicted_sampling_MSE':variance,'programmed_mean_bias_MSE':bias_mse,'predicted_target_MSE':variance+bias_mse,'observed_target_MSE':float(np.mean((y.astype(float)-target)**2)),'observed_MSE_vs_programmed_mean':float(np.mean((y.astype(float)-mean)**2))}
        for name,value in values.items():np.testing.assert_allclose(meta[name],value,atol=1e-13,rtol=1e-11)
        assert meta['observed_node_updates']==meta['logical_weight_readouts']==t*n and meta['binary_product_terms']==64*t*n and meta['free_nodes']==n and meta['clamped_nodes']==0 and meta['bank_batches']==1
        rows.append({'condition':case['condition'],'stage':stage,'seed':seed,'group':g,'samples':t,'public_F32_output_bitwise_exact':True,'integer_counts_sha256':digest(counts),'scaled_F64_frequency_error':scaled,'descriptive_2e_5_passed':bool(scaled<2e-5),'max_error_over_IEEE_bound':float(np.max(error/np.maximum(bound,1e-30))),'production_updates_checked':t*n,'native_replay_updates':2*t*n});print(f'Independent native range replay stage{stage} {case["condition"]} seed{seed}: exact output',flush=True)
    assert len(rows)==360 and {(r['condition'],r['stage'],r['seed']) for r in rows}=={(c['name'],k,s) for c in p['native_conditions'] for k in range(24) for s in range(5)};assert sum(r['production_updates_checked'] for r in rows)==m['native_observed_updates']
    r={'passed':True,'native_operators':360,'every_public_F32_output_bitwise_exact':True,'all_frequency_IEEE_envelopes_verified':True,'descriptive_2e_5_failures_retained':sum(not r['descriptive_2e_5_passed'] for r in rows),'native_replay_observed_updates':sum(r['native_replay_updates'] for r in rows),'source_sha256':sha(__file__),'rows':rows,'scope':'Full-size stock THRML unary operators replayed with independent public factors/observers and integer frequencies, dynamic fields/defaultF32. All exact outputs, targets, resources and derived forward-error bounds checked;2e-5 flag descriptive only. Actual neural anchor inputs checked in separate original-model audit. No full stochastic neural quality or physical advantage claimed.'};atomic_json(OUT/'native_evidence_audit.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
