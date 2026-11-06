"""Independent public direct sums, integer count reconstruction and error bounds."""
import os,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state,FactorSamplingProgram
from thrml.models.discrete_ebm import SpinGibbsConditional,SpinEBMFactor
from .calibrated_mnist_path_audit import PublicLayer
from .positional_operator_common import OUT,sha,guard,digest

class PublicPositionalSum(AbstractObserver):
    readout:object
    x:jax.Array
    scale:jax.Array
    shape:tuple
    k:int
    def init(self):return jnp.zeros((self.x.shape[0],self.shape[1]),jnp.float32)
    def __call__(self,program,free,clamped,total,iteration):
        global_state=block_state_to_global(free+clamped,program.gibbs_spec);flat=from_global_state(global_state,program.gibbs_spec,[self.readout])[0];planes=flat.reshape(self.shape+(self.k,)).astype(jnp.float32);code=jnp.sum(planes*jnp.asarray([2**j for j in range(self.k)],jnp.float32),axis=-1);weight=code/((1<<self.k)-1)
        return total+self.x@(weight*self.scale),None

def build(layer,shape,k,t):
    def run(key,x,fields,lower,scale,bias):
        program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,fields)],[]);obs=PublicPositionalSum(layer.readout,x,scale,shape,k);total,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),layer.initial,[],obs.init(),obs);return total/t+x@lower+bias
    return jax.jit(run)

def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';p=json.loads((OUT/'protocol.json').read_text());guard(p);r=json.loads((OUT/'statistics.json').read_text());assert r['protocol_sha256']==sha(OUT/'protocol.json');layers={};functions={};rows=[];updates=0;u=2.**-24
    for case in r['rows']:
        sig=case['signature'];assert sig['protocol_sha256']==sha(OUT/'protocol.json');stage,k,t,seed=[sig[key] for key in ['stage','K','T','seed']];entry=p['matrices'][stage];c=entry['single_controls' if k==1 else 'positional_controls'];assert sha(c['path'])==c['sha256']==sig['controls_sha256'];a=np.load(c['path']);shape=tuple(a['codes'].shape);lo=np.repeat(a['lower'],16,0);sp=np.repeat(a['span'],16,0);bias=a['bias'];fields=((-4+a['codes'].astype(float)*8/255).astype(np.float32)[...,None] if k==1 else np.where(((a['codes'][...,None]//(2**np.arange(4)))%2),4,-4).astype(np.float32));assert digest(fields)==sig['fields_sha256'];anchor=entry['anchor'];assert sha(anchor['inputs'])==sig['inputs_sha256'] and sha(anchor['target'])==sig['target_sha256'];x=np.load(anchor['inputs']);target=np.load(anchor['target']);path=Path(case['output']);assert sha(path)==case['output_sha256'];y=np.load(path);cache=(shape,k)
        if cache not in layers:layers[cache]=PublicLayer(np.full(shape+(k,),.5));assert layers[cache].n_free==int(np.prod(shape))*k and not layers[cache].clamped
        layer=layers[cache];key=jax.random.key(p['key_root'])
        for v in [stage,k,seed]:key=jax.random.fold_in(key,v)
        if (cache,t) not in functions:functions[cache,t]=build(layer,shape,k,t)
        xx,ff,ll,ss,bb=[jnp.asarray(z) for z in [x,fields.ravel(),lo,sp,bias]];replay=np.asarray(functions[cache,t](key,xx,ff,ll,ss,bb));exact=bool(np.array_equal(replay,y));frequency=np.asarray(layer.frequencies(t)(key[None],ff))[0];counts=np.rint(frequency.astype(float)*t).astype(np.int32);assert np.all((counts>=0)&(counts<=t));coeff=np.array([2**j for j in range(k)],float)/((1<<k)-1);average=np.sum(counts.astype(float)*coeff,axis=-1)/t;reference=x.astype(float)@(lo.astype(float)+sp.astype(float)*average)+bias;error=np.abs(y.astype(float)-reference)
        d=shape[0];gamma=lambda n:n*u/(1-n*u);gd,gt,g4=gamma(2*d+4),gamma(t-1),gamma(4);mag=np.abs(x.astype(float))@np.abs(sp.astype(float));center=np.abs(x.astype(float))@np.abs(lo.astype(float));ab=np.abs(bias.astype(float));bound=(gd+gt*(1+gd))*mag+gd*center+g4*((1+gt)*(1+gd)*mag+(1+gd)*center+ab)+(4*d+2*t+16)*np.finfo(np.float32).tiny+((4*d+16)*2.**-53)*(mag+center+ab);ieee=bool(np.all(error<=bound));assert np.max(t*(mag+center+ab))<np.finfo(np.float32).max
        q=1/(1+np.exp(-2*fields.astype(float)));mean=np.sum(q*coeff,axis=-1);variance=np.sum(q*(1-q)*coeff**2,axis=-1);ym=x.astype(float)@(lo.astype(float)+sp.astype(float)*mean)+bias;pred=float(np.mean(x.astype(float)**2@(sp.astype(float)**2*variance))/t);bias_mse=float(np.mean((ym-target)**2));metrics={'predicted_sampling_MSE':pred,'programmed_mean_bias_MSE':bias_mse,'predicted_target_MSE':pred+bias_mse,'observed_target_MSE':float(np.mean((y.astype(float)-target)**2)),'observed_MSE_vs_programmed_mean':float(np.mean((y.astype(float)-ym)**2))}
        for name,value in metrics.items():np.testing.assert_allclose(case[name],value,rtol=1e-10,atol=1e-13)
        n=int(np.prod(shape));assert case['native_node_updates']==n*k*t and case['free_nodes']==n*k and case['clamped_nodes']==0 and case['binary_input_terms']==64*k*n*t and case['converted_control_range_bias_payload_bits']==n*(8 if k==1 else 4)+64*(n//16)+32*len(bias)
        row={'case':case['case'],'public_F32_exact':exact,'IEEE_frequency_envelope_passed':ieee,'max_public_difference':float(np.max(np.abs(replay.astype(float)-y.astype(float)))),'counts_sha256':digest(counts),'max_error_over_IEEE_bound':float(np.max(error/np.maximum(bound,1e-30))),'native_replay_updates':2*n*k*t};rows.append(row);updates+=2*n*k*t
        if seed==p['seeds'][-1]:print(f'Public direct-sum/count replay stage{stage} K{k} T{t}: exact={all(v["public_F32_exact"] for v in rows[-5:])}',flush=True)
    assert len(rows)==960 and updates==2*r['native_node_updates'];guard(p);result={'completed':True,'passed':all(v['public_F32_exact'] and v['IEEE_frequency_envelope_passed'] for v in rows),'native_operators':len(rows),'all_public_F32_exact':all(v['public_F32_exact'] for v in rows),'all_IEEE_frequency_envelopes_passed':all(v['IEEE_frequency_envelope_passed'] for v in rows),'independent_native_replay_updates':updates,'producer_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'rows':rows,'scope':'Independent public sum-only and separate integer-frequency paths for every operator, metrics/resources and finite F32 IEEE envelopes. Validation teacher activations; no end-to-end neural or hardware claim.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert result['passed'],'Frozen exact/envelope gate failed; preserve all failures.'
if __name__=='__main__':main()
