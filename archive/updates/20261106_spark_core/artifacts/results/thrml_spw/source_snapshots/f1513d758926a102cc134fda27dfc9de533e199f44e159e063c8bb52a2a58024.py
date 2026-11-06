"""Actual learned positive layer with native activation/weight AND-count execution."""
from pathlib import Path
import json,hashlib,time
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state
from .native import NativeProbabilityBank
from .controlled_positive_native import set_fields
OUT=Path('results/thrml_spw/boolean_layer')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def field(p):
    eps=1/(1+np.exp(8));q=np.clip(np.asarray(p,float),eps,1-eps);f=.5*(np.log(q)-np.log1p(-q));code=np.rint((np.clip(f,-4,4)+4)*255/8);return (-4+8*code/255).astype(np.float32)
class AndCounts(AbstractObserver):
    readout:object
    def init(self):return jnp.zeros((8,32),jnp.int32),jnp.zeros((64,32),jnp.int32)
    def __call__(self,program,free,clamped,carry,iteration):
        state=block_state_to_global(free+clamped,program.gibbs_spec);z=from_global_state(state,program.gibbs_spec,[self.readout])[0];w=z[:2048].reshape(64,32);a=z[2048:].reshape(8,64);and_sum=jnp.sum(a[:,:,None]&w[None,:,:],axis=1,dtype=jnp.int32);return (carry[0]+and_sum,carry[1]+w.astype(jnp.int32)),None

def main():
    assert jax.default_backend()=='cpu';assert not (OUT/'manifest.json').exists();OUT.mkdir(parents=True,exist_ok=True);inputs=Path('data/thrml_spw/positive_mnist/evaluation_data.npz');models=[Path(f'data/thrml_spw/positive_mnist/seed{s}.npz') for s in range(5)];sources={str(Path('research/thrml_spw')/s):sha(Path('research/thrml_spw')/s) for s in ['boolean_layer.py','verify_boolean_layer.py','boolean_layer_report.py','native.py','controlled_positive_native.py']};p={'model_seeds':list(range(5)),'sampling_seeds':list(range(5)),'trials_per_seed':200,'chain_batch':50,'samples':[1,4,16,64,256],'input_source':str(inputs),'input_sha256':sha(inputs),'model_sources':{str(f):sha(f) for f in models},'key_root':1267520,'field_bound':4,'field_precision_bits':8,'input_selection':'first8 validation inputs, no labels/task scores','source_sha256':sources,'shape':[64,32],'activation_shape':[8,64],'all_free_nodes':2560,'weight_nodes':2048,'activation_nodes':512,'statistical_gate':'Every estimator/coordinate mean must satisfy finite-sample Bernstein bound at familyalpha.01 over25conditions*3estimators*256outputs; variance/covariance comparisons descriptive. Independent public state replay must reproduce every AND/weight integer count exactly.','scope':'First learned positive-weight layer of five fixed MNIST models;8 consumed validation examples, no classifier/no new neural task score. Native AND plus integer accumulation only for AND path; F64 host reconstruction and float-activation controls are analysis. IID finite8-bit fields, no physical gate/energy/latency claim.'};(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');x=np.load(inputs)['validation_inputs'][:8];bank=NativeProbabilityBank(np.full(2560,.5));initial=[jnp.zeros(2560,jnp.bool_)];start=time.monotonic();rows=[];updates=0
    @lru_cache(None)
    def kernel(t):
        def batch(keys,fields):
            program=set_fields(bank,fields)
            def one(key):
                obs=AndCounts(bank.readout);result,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),initial,[],obs.init(),obs);return result
            return jax.vmap(one)(keys)
        return jax.jit(batch)
    for model,path in enumerate(models):
        a=np.load(path);w=a['weight_0'];bias=a['bias_0'];fields=np.concatenate([field(w).ravel(),field(x).ravel()]);q=1/(1+np.exp(-2*fields.astype(float)));qw,qa=q[:2048].reshape(64,32),q[2048:].reshape(8,64);target=x.astype(float)@w.astype(float)+bias
        control=OUT/f'model{model}_controls.npz';np.savez_compressed(control,fields=fields,weight=w,bias=bias,inputs=x,programmed_weight=qw,programmed_activation=qa)
        for t in p['samples']:
            raw_and=np.empty((5,200,8,32),np.int32);raw_w=np.empty((5,200,64,32),np.int32)
            for si,seed in enumerate(p['sampling_seeds']):
                for first in range(0,200,50):
                    key=jax.random.key(p['key_root'])
                    for value in [model,seed,t,first]:key=jax.random.fold_in(key,value)
                    aa,ww=kernel(t)(jax.random.split(key,50),jnp.asarray(fields));raw_and[si,first:first+50]=np.asarray(aa);raw_w[si,first:first+50]=np.asarray(ww);updates+=50*t*2560
            path_out=OUT/f'model{model}_T{t}.npz';np.savez_compressed(path_out,and_counts=raw_and,weight_counts=raw_w);estimators={'AND':raw_and.astype(float)/t+bias,'matched_float':np.einsum('mi,srio->srmo',qa,raw_w.astype(float)/t)+bias,'requested_float':np.einsum('mi,srio->srmo',x.astype(float),raw_w.astype(float)/t)+bias};means={'AND':qa@qw+bias,'matched_float':qa@qw+bias,'requested_float':x.astype(float)@qw+bias};variances={'AND':(qa@qw-qa**2@qw**2)/t,'matched_float':qa**2@(qw*(1-qw))/t,'requested_float':x.astype(float)**2@(qw*(1-qw))/t};logfactor=np.log(2*25*3*256/.01);details={}
            for name,y in estimators.items():
                error=y-means[name];mean_error=error.mean((0,1));bound=np.sqrt(2*variances[name]*logfactor/1000)+64*logfactor/(3*1000);details[name]={'observed_sampling_MSE':float(np.mean(error**2)),'predicted_sampling_MSE':float(variances[name].mean()),'observed_target_MSE':float(np.mean((y-target)**2)),'predicted_target_MSE':float(np.mean(variances[name]+(means[name]-target)**2)),'all_coordinate_means_within_Bernstein':bool(np.all(np.abs(mean_error)<=bound)),'max_mean_error_over_Bernstein':float(np.max(np.abs(mean_error)/bound)),'seed_sampling_MSE':np.mean(error**2,axis=(1,2,3)).tolist()}
            extra=qa*(1-qa)@qw/t;np.testing.assert_allclose(variances['AND']-variances['matched_float'],extra,rtol=1e-11,atol=1e-14);cov=qw.T@(qa[0,:,None]*(1-qa[0,:,None])*qw)/t;np.fill_diagonal(cov,variances['AND'][0]);observed=np.cov(estimators['AND'].reshape(1000,8,32)[:,0],rowvar=False,ddof=1);np.savez_compressed(OUT/f'model{model}_T{t}_covariance.npz',theory=cov,observed=observed);row={'model':model,'T':t,'raw':str(path_out),'raw_sha256':sha(path_out),'controls':str(control),'controls_sha256':sha(control),'native_node_updates':1000*t*2560,'logical_AND_terms':1000*t*8*64*32,'estimators':details,'predicted_extra_activation_variance':float(extra.mean()),'covariance':str(OUT/f'model{model}_T{t}_covariance.npz'),'covariance_sha256':sha(OUT/f'model{model}_T{t}_covariance.npz')};rows.append(row);print(f'Boolean learned layer model{model} T{t}: native counts saved elapsed={time.monotonic()-start:.1f}s',flush=True)
    for source,h in sources.items():assert sha(source)==h
    assert updates==4364800000;result={'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'cases':25,'native_node_updates':updates,'all_mean_Bernstein_checks_passed':all(v['all_coordinate_means_within_Bernstein'] for row in rows for v in row['estimators'].values()),'rows':rows,'elapsed_seconds':time.monotonic()-start,'new_task_scores':0,'scope':p['scope']};(OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert result['all_mean_Bernstein_checks_passed']
if __name__=='__main__':main()
