"""Independent public THRML reconstruction, trajectory and scalar moment audit."""
import json,math
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,BlockGibbsSpec,FactorSamplingProgram,SamplingSchedule,SpinNode,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .finite_positional_common import OUT,sha,source_guard

def sampler(fields,t,n):
    nodes=[SpinNode() for _ in fields.ravel()];block=Block(nodes)
    spec=BlockGibbsSpec([block],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)})
    program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([block],jnp.asarray(fields.ravel()))],[])
    schedule=SamplingSchedule(n_warmup=1,n_samples=t,steps_per_sample=1)
    initial=[jnp.zeros(len(nodes),dtype=jnp.bool_)]
    def run(key):return sample_states(key,program,schedule,initial,[],[block])[0]
    return jax.jit(jax.vmap(run))

def main():
    p=json.loads((OUT/'protocol.json').read_text());source_guard(p);r=json.loads((OUT/'statistics.json').read_text());assert r['protocol_sha256']==sha(OUT/'protocol.json') and jax.default_backend()=='gpu';checked=0;updates=0;used_keys=set();moments_checked=0
    for path,h in r['raw_sha256'].items():
        assert sha(path)==h;raw=np.load(path);bound=int(path.split('/')[-1].split('_')[0][1:]);k=raw['counts'].shape[-1];target=np.linspace(0,1,257);np.testing.assert_array_equal(raw['targets'],target);budgets=raw['budgets'];fields=raw['fields'];coeff=np.array([2**i/((1<<k)-1) for i in range(k)])
        if k==1:
            prob=np.clip(target,1/(1+math.exp(2*bound)),1/(1+math.exp(-2*bound)));continuous=np.clip(.5*np.log(prob/(1-prob)),-bound,bound);code=np.rint((continuous+bound)*255/(2*bound)).astype(np.uint8);expected=(-bound+code.astype(float)*2*bound/255).astype(np.float32)[:,None];np.testing.assert_array_equal(raw['codes'],code)
        else:
            integer=np.rint(target*((1<<k)-1)).astype(int);code=((integer[:,None]//(2**np.arange(k)))%2).astype(np.uint8);expected=np.where(code,bound,-bound).astype(np.float32);np.testing.assert_array_equal(raw['codes'],code)
        np.testing.assert_array_equal(fields,expected);np.testing.assert_array_equal(coeff,raw['coefficients']);q=np.array([[1/(1+math.exp(-2*float(f))) for f in row] for row in fields]);np.testing.assert_allclose(q,raw['probabilities'],rtol=1e-14,atol=1e-16)
        run=sampler(fields,int(max(budgets)),p['chain_batch'])
        for si,seed in enumerate(p['seeds']):
            for first in range(0,p['trials_per_seed'],p['chain_batch']):
                key_int=p['key_root']+bound*1000000+k*100000+seed*1000+first;assert key_int not in used_keys;used_keys.add(key_int);n=min(p['chain_batch'],p['trials_per_seed']-first);keys=jax.random.split(jax.random.key(key_int),n);bits=np.asarray(run(keys)).reshape(n,int(max(budgets)),len(target),k);counts=np.cumsum(bits,axis=1,dtype=np.uint16)[:,budgets-1];np.testing.assert_array_equal(counts,raw['counts'][si,first:first+n]);checked+=counts.size;updates+=n*int(max(budgets))*len(target)*k
            print(f'Independent public replay {path} seed={seed}: exact',flush=True)
        # Convolve raw moments of independent centered binomial sample means.
        for ti,t in enumerate(budgets):
            mean=q@coeff;v=np.zeros(len(target));third=np.zeros_like(v);fourth=np.zeros_like(v)
            for bit,a in enumerate(coeff):
                z=q[:,bit];vv=a*a*z*(1-z)/t;mm3=a**3*z*(1-z)*(1-2*z)/t**2;mm4=a**4*(3*(z*(1-z))**2/t**2+z*(1-z)*(1-6*z*(1-z))/t**3)
                fourth=fourth+6*v*vv+mm4;third+=mm3;v+=vv
            b=mean-target;theory=float(np.mean(v+b*b));square_variance=4*b*b*v+4*b*third+fourth-v*v;se=float(np.sqrt(np.maximum(square_variance,0).sum()/(1000*len(target)**2)));estimate=np.sum(raw['counts'][:,:,ti].astype(float)*coeff,axis=-1)/t;error=estimate-target;mse=float(np.mean(error**2));row=next(x for x in r['rows'] if x['field_bound']==bound and x['pbits_per_scalar']==k and x['samples']==t)
            for key,val in {'mse':mse,'rmse':np.sqrt(mse),'theory_mse':theory,'mse_standard_error':se,'mse_z':(mse-theory)/se,'mean_bias':error.mean(),'mae':np.abs(error).mean(),'node_updates_per_scalar_trial':k*t,'theory_variance':v.mean(),'theory_squared_bias':(b*b).mean()}.items():np.testing.assert_allclose(row[key],val,rtol=2e-7,atol=1e-12)
            moments_checked+=1
    assert updates==r['native_node_updates'] and r['passed']==all(abs(x['mse_z'])<6 for x in r['rows']);source_guard(p)
    result={'passed':True,'public_counts_exact':checked,'unique_key_batches':len(used_keys),'analytic_and_empirical_rows_checked':moments_checked,'independent_native_node_updates':updates,'all_free_nodes_no_clamps':True,'original_scalar_gate_passed':r['passed'],'producer_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'scope':'Exact independent public THRML replay and centered-moment audit. No neural or physical result.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
