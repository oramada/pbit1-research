"""Independent public state replay, moment recursion and full-control activity audit."""
from pathlib import Path
from functools import lru_cache
import json,math
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,SpinNode,BlockGibbsSpec,FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .state_activity_math import OUT,sha

def recursive_moments(p,t,retained):
    probability=np.stack([1-p,p]) if retained else np.stack([np.ones_like(p),np.zeros_like(p)]);first=np.zeros_like(probability);second=np.zeros_like(probability)
    for _ in range(t):
        np0=np.zeros_like(probability);nm=np.zeros_like(first);ns=np.zeros_like(second)
        for a in range(2):
            for b in range(2):
                delta=int(a!=b);weight=p if b else 1-p;np0[b]+=probability[a]*weight;nm[b]+=(first[a]+delta*probability[a])*weight;ns[b]+=(second[a]+2*delta*first[a]+delta*probability[a])*weight
        probability,first,second=np0,nm,ns
    mean=first.sum(0);return mean,np.maximum(second.sum(0)-mean**2,0)

def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled;protocol=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());ledger=json.loads((OUT/'ledger.json').read_text());assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['ledger_sha256']==sha(OUT/'ledger.json')
    for source,h in protocol['source_sha256'].items():assert sha(source)==h
    block=Block([SpinNode() for _ in range(256)]);spec=BlockGibbsSpec([block],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)});initial=[jnp.zeros(256,jnp.bool_)];counter_elements=0;updates=0;gate_checks=0
    @lru_cache(None)
    def sampler(t):
        def batch(keys,fields):
            program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([block],fields)],[])
            def one(key):return sample_states(key,program,SamplingSchedule(1,t+1,1),initial,[],[block])[0]
            return jax.vmap(one)(keys)
        return jax.jit(batch)
    for case in m['cases']:
        assert sha(case['path'])==case['sha256'];a=np.load(case['path']);t=case['T'];fields=np.array([np.float32(8*c/255-4) for c in range(256)]);np.testing.assert_array_equal(fields,a['fields']);p=np.array([1/(1+math.exp(-2*float(b))) for b in fields]);np.testing.assert_allclose(p,a['probabilities'],rtol=1e-14,atol=1e-16)
        for si,seed in enumerate(range(5)):
            for first in range(0,200,50):
                key=jax.random.key(protocol['key_root'])
                for value in [seed,t,first]:key=jax.random.fold_in(key,value)
                states=np.asarray(sampler(t)(jax.random.split(key,50),jnp.asarray(fields)));left=np.concatenate([np.zeros((50,1,256),bool),states[:,:t-1]],axis=1);right=states[:,:t];before,after=states[:,:-1],states[:,1:];expected={'last':states[:,-1],'first':states[:,0],'reset_last':states[:,t-1],'reset_flips':(left!=right).sum(1,dtype=np.int32),'reset_up':((~left)&right).sum(1,dtype=np.int32),'reset_down':(left&(~right)).sum(1,dtype=np.int32),'retained_flips':(before!=after).sum(1,dtype=np.int32),'retained_up':((~before)&after).sum(1,dtype=np.int32),'retained_down':(before&(~after)).sum(1,dtype=np.int32)}
                for name,value in expected.items():np.testing.assert_array_equal(value,a[name][si,first:first+50]);counter_elements+=value.size
                updates+=50*(t+1)*256
        for mode,retained in [('reset',False),('retained',True)]:
            mean,var=recursive_moments(p,t,retained);observed=a[mode+'_flips'].mean((0,1));L=np.log(2*2560/.01);bound=np.sqrt(2*var*L/1000)+t*L/3000;passed=bool(np.all(np.abs(observed-mean)<=bound));assert passed==case['checks'][mode]['all_means_within_Bernstein'];np.testing.assert_allclose(case['checks'][mode]['max_mean_error_over_bound'],np.max(np.abs(observed-mean)/bound),rtol=2e-8,atol=1e-11);gate_checks+=256
        print(f'Independent public activity T{t}: all9 state/count arrays exact, recursive moments checked',flush=True)
    assert updates==m['native_node_updates']==88576000 and gate_checks==2560;expected_keys=set();full_rows=[]
    for entry in protocol['matrices']:
        n=entry['weights']
        for k,key in [(1,'single_controls'),(4,'positional_controls')]:
            control=entry[key];assert sha(control['path'])==control['sha256'];codes=np.load(control['path'])['codes'];assert codes.size==n
            fields=(-4+codes.astype(float)*8/255).astype(np.float32) if k==1 else np.where(((codes[...,None]//(2**np.arange(4)))%2),4.,-4.).astype(np.float32);p=1/(1+np.exp(-2*fields.astype(float)));variance=p*(1-p)
            for budget in [4,16,64,256]:
                t=budget//k;row=next(r for r in ledger['rows'] if r['stage']==entry['stage'] and r['K']==k and r['updates_per_weight']==budget);expected_keys.add((entry['stage'],k,budget));reset=float(np.sum(p+2*(t-1)*variance));retained=float(np.sum(2*t*variance));values={'weights':n,'logical_free_nodes':n*k,'samples':t,'scheduled_updates_per_bank':budget*n,'expected_ones_in_first_record':p.sum(),'sum_p_one_minus_p':variance.sum(),'expected_reset_changes_per_bank':reset,'expected_retained_changes_per_bank':retained,'expected_reset_0_to_1':np.sum(p+(t-1)*variance),'expected_reset_1_to_0':np.sum((t-1)*variance),'expected_retained_0_to_1':np.sum(t*variance),'expected_retained_1_to_0':np.sum(t*variance)}
                for name,value in values.items():np.testing.assert_allclose(row[name],value,rtol=2e-12,atol=1e-6)
                full_rows.append({'budget':budget,'K':k,'reset':reset,'retained':retained,'weights':n})
    assert len(ledger['rows'])==len(expected_keys)==192
    for row in ledger['totals']:
        selected=[r for r in full_rows if r['budget']==row['updates_per_weight'] and r['K']==row['K']];n=sum(r['weights'] for r in selected);reset=sum(r['reset'] for r in selected);retained=sum(r['retained'] for r in selected)
        for name,value in {'weights':n,'samples':row['updates_per_weight']//row['K'],'scheduled_updates_per32_context_forward':4*row['updates_per_weight']*n,'expected_reset_changes_per32_context_forward':4*reset,'expected_retained_changes_per32_context_forward':4*retained,'reset_changes_per_weight_per_bank':reset/n,'retained_changes_per_weight_per_bank':retained/n}.items():np.testing.assert_allclose(row[name],value,rtol=2e-12,atol=1e-6)
    result={'passed':True,'public_counter_and_state_elements_exact':counter_elements,'native_witness_cases':5,'mean_gate_conditions_checked':gate_checks,'full_control_ledger_rows_checked':192,'aggregate_rows_checked':8,'independent_native_updates':updates,'producer_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'scope':'Discrete recorded-state proxy, not measured physical flips/energy. Full learned fields analytically checked; separate native controller-code witness exactly replayed.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
