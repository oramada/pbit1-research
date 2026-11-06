"""Independent public raw-state replay of every learned-layer integer count."""
from functools import lru_cache
import json,math
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,BlockGibbsSpec,SpinNode,FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .boolean_layer import OUT,sha

def encode(value):
    eps=1/(1+math.exp(8));q=np.clip(value.astype(float),eps,1-eps);f=np.clip(np.log(q/(1-q))/2,-4,4);code=np.rint((f+4)*255/8);return (-4+code*8/255).astype(np.float32)

def main():
    assert jax.default_backend()=='cpu';p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['protocol_sha256']==sha(OUT/'protocol.json')
    for source,h in p['source_sha256'].items():assert sha(source)==h
    assert sha(p['input_source'])==p['input_sha256'];x=np.load(p['input_source'])['validation_inputs'][:8];nodes=[SpinNode() for _ in range(2560)];block=Block(nodes);spec=BlockGibbsSpec([block],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)});initial=[jnp.zeros(2560,jnp.bool_)];updates=0;count_elements=0
    @lru_cache(None)
    def raw_states(t):
        def batch(keys,fields):
            program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([block],fields)],[])
            def one(key):return sample_states(key,program,SamplingSchedule(1,t,1),initial,[],[block])[0]
            return jax.vmap(one)(keys)
        return jax.jit(batch)
    for row in m['rows']:
        model,t=row['model'],row['T'];source=f'data/thrml_spw/positive_mnist/seed{model}.npz';assert sha(source)==p['model_sources'][source];original=np.load(source);w,bias=original['weight_0'],original['bias_0'];assert np.all((w>=0)&(w<=1)) and np.all((x>=0)&(x<=1));assert sha(row['raw'])==row['raw_sha256'] and sha(row['controls'])==row['controls_sha256'];a=np.load(row['controls']);fields=np.concatenate([encode(w).ravel(),encode(x).ravel()]);np.testing.assert_array_equal(fields,a['fields']);np.testing.assert_array_equal(w,a['weight']);np.testing.assert_array_equal(x,a['inputs']);np.testing.assert_array_equal(bias,a['bias']);raw=np.load(row['raw']);counts=raw['and_counts'];wc=raw['weight_counts'];assert counts.dtype==wc.dtype==np.int32
        for si,seed in enumerate(range(5)):
            for first in range(0,200,50):
                key=jax.random.key(p['key_root'])
                for value in [model,seed,t,first]:key=jax.random.fold_in(key,value)
                states=np.asarray(raw_states(t)(jax.random.split(key,50),jnp.asarray(fields)));wb=states[:,:,:2048].reshape(50,t,64,32);ab=states[:,:,2048:].reshape(50,t,8,64);expected_w=wb.sum(1,dtype=np.int32);expected_and=np.einsum('btmi,btio->bmo',ab.astype(np.int32),wb.astype(np.int32),dtype=np.int32,optimize=False);np.testing.assert_array_equal(expected_w,wc[si,first:first+50]);np.testing.assert_array_equal(expected_and,counts[si,first:first+50]);updates+=50*t*2560;count_elements+=expected_w.size+expected_and.size
        probability=np.array([1/(1+math.exp(-2*float(v))) for v in fields]);qw=probability[:2048].reshape(64,32);qa=probability[2048:].reshape(8,64);np.testing.assert_allclose(qw,a['programmed_weight'],rtol=1e-14,atol=1e-16);np.testing.assert_allclose(qa,a['programmed_activation'],rtol=1e-14,atol=1e-16);target=x.astype(float)@w.astype(float)+bias;expected=qa@qw+bias;variance_and=np.zeros((8,32));extra=np.zeros((8,32))
        for i in range(64):
            product=qa[:,i,None]*qw[i,None,:];variance_and+=product*(1-product)/t;extra+=qa[:,i,None]*(1-qa[:,i,None])*qw[i,None,:]/t
        matched=np.einsum('mi,srio->srmo',qa,wc.astype(float)/t)+bias;requested=np.einsum('mi,srio->srmo',x.astype(float),wc.astype(float)/t)+bias;ys={'AND':counts.astype(float)/t+bias,'matched_float':matched,'requested_float':requested};means={'AND':expected,'matched_float':expected,'requested_float':x.astype(float)@qw+bias};variances={'AND':variance_and,'matched_float':qa**2@(qw*(1-qw))/t,'requested_float':x.astype(float)**2@(qw*(1-qw))/t};np.testing.assert_allclose(row['predicted_extra_activation_variance'],extra.mean(),rtol=1e-11,atol=1e-13)
        for name,y in ys.items():
            error=y-means[name];bound=np.sqrt(2*variances[name]*np.log(2*19200/.01)/1000)+64*np.log(2*19200/.01)/3000;passed=bool(np.all(np.abs(error.mean((0,1)))<=bound));values={'observed_sampling_MSE':np.mean(error**2),'predicted_sampling_MSE':variances[name].mean(),'observed_target_MSE':np.mean((y-target)**2),'predicted_target_MSE':np.mean(variances[name]+(means[name]-target)**2),'max_mean_error_over_Bernstein':np.max(np.abs(error.mean((0,1)))/bound)}
            for key,value in values.items():np.testing.assert_allclose(row['estimators'][name][key],value,rtol=1e-11,atol=1e-13)
            assert passed==row['estimators'][name]['all_coordinate_means_within_Bernstein'];np.testing.assert_allclose(row['estimators'][name]['seed_sampling_MSE'],np.mean(error**2,axis=(1,2,3)),rtol=1e-11,atol=1e-13)
        assert sha(row['covariance'])==row['covariance_sha256'];cov=np.load(row['covariance']);theory=np.empty((32,32))
        for o in range(32):
            for v in range(32):theory[o,v]=variance_and[0,o] if o==v else np.sum(qw[:,o]*qw[:,v]*qa[0]*(1-qa[0]))/t
        np.testing.assert_allclose(cov['theory'],theory,rtol=1e-11,atol=1e-13);z=ys['AND'].reshape(1000,8,32)[:,0];centered=z-z.mean(0);np.testing.assert_allclose(cov['observed'],centered.T@centered/999,rtol=1e-11,atol=1e-13);assert row['native_node_updates']==1000*2560*t and row['logical_AND_terms']==1000*t*8*64*32;print(f'Independent public boolean layer model{model} T{t}: every integer count, mean/variance/covariance exact',flush=True)
    assert updates==m['native_node_updates']==4364800000;assert m['all_mean_Bernstein_checks_passed'];result={'passed':True,'learned_layer_cases':25,'integer_count_elements_reproduced_exactly':count_elements,'independent_public_native_node_updates':updates,'all_means_variances_covariances_resources_checked':True,'producer_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'scope':'Independent public THRML raw states and integer products reproduce all AND/weight counts and F64 analyses. Learned first layer only; no new classifier accuracy or measured hardware result.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
