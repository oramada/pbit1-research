"""Bounded witness using separate public raw-state sampling, not task scoring."""
import hashlib,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,BlockGibbsSpec,SpinNode,FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .boolean_classifier_native import CountLayer,readout_boolean,readout_weight
OUT=Path('results/thrml_spw/boolean_classifier_native_witness')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_sampler(n,t):
    b=Block([SpinNode() for _ in range(n)]);spec=BlockGibbsSpec([b],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)})
    def run(key,fields):
        program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([b],fields)],[])
        return sample_states(key,program,SamplingSchedule(1,t,1),[jnp.zeros(n,jnp.bool_)],[],[b])[0]
    return jax.jit(jax.vmap(run))
def main():
    assert not (OUT/'evidence_audit.json').exists();OUT.mkdir(exist_ok=True)
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled
    sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['boolean_classifier_native.py','verify_boolean_classifier_native.py','native.py','controlled_positive_native.py']}
    # Includes maximum prospective Boolean T with high-probability nodes to expose uint16 overflow.
    cases=[(64,32,False,7),(32,10,False,7),(64,32,True,7),(32,10,True,7),(64,32,True,1184)]
    (OUT/'protocol.json').write_text(json.dumps({'sources':sources,'cases':cases,'chains':2,'key_root':1317520,'scope':'Synthetic count/readout witness, no classifier accuracy. Both observer and replay scheduled updates charged.'},indent=2)+'\n')
    rows=[];updates=0;elements=0
    for ci,(ni,no,boolean,t) in enumerate(cases):
        layer=CountLayer(ni,no,8,boolean);n=layer.n
        fields=np.full((2,n),4,np.float32) if t==1184 else np.linspace(-4,4,2*n,dtype=np.float32).reshape(2,n)
        keys=jax.random.split(jax.random.fold_in(jax.random.key(1317520),ci),2)
        observed=layer.kernel(t)(keys,jnp.asarray(fields));raw=np.asarray(raw_sampler(n,t)(keys,jnp.asarray(fields)));w=raw[:,:,:ni*no].reshape(2,t,ni,no)
        if boolean:
            activation=raw[:,:,ni*no:].reshape(2,t,8,ni)
            want_and=np.einsum('ctbi,ctio->cbo',activation.astype(np.int32),w.astype(np.int32),dtype=np.int32,optimize=False)
            want_col=w.sum((1,2),dtype=np.int32)
            aa,cc=map(np.asarray,observed);np.testing.assert_array_equal(aa,want_and);np.testing.assert_array_equal(cc,want_col)
            lower=-.4242129623889923;span=3.25;bias=np.linspace(-1,1,no)
            y=readout_boolean(aa,cc[:,None,:],t,lower,span,bias)
            direct=np.einsum('ctbi,ctio->cbo',lower+span*activation.astype(float),w.astype(float),optimize=False)/t+bias
            np.testing.assert_allclose(y,direct,rtol=1e-12,atol=1e-11)
            if t==1184:assert int(aa.max())>65535 and int(cc.max())>65535
            np.savez_compressed(OUT/f'case{ci}.npz',fields=fields,and_counts=aa,column_counts=cc,readout=y)
            count=aa.size+cc.size
        else:
            actual=np.asarray(observed).reshape(2,ni,no);want=w.sum(1,dtype=np.int32);np.testing.assert_array_equal(actual,want)
            x=np.linspace(-.5,2.5,8*ni).reshape(8,ni);bias=np.linspace(-1,1,no)
            y=readout_weight(actual,t,x,bias)
            direct=np.einsum('bi,ctio->cbo',x,w.astype(float),optimize=False)/t+bias
            np.testing.assert_allclose(y,direct,rtol=1e-12,atol=1e-11)
            np.savez_compressed(OUT/f'case{ci}.npz',fields=fields,weight_counts=actual,readout=y)
            count=actual.size
        elements+=count;updates+=2*t*n
        rows.append({'case':ci,'ni':ni,'no':no,'boolean':boolean,'T':t,'counter_entries_exact':count,'native_observer_updates':2*t*n,'independent_raw_updates':2*t*n})
        print(json.dumps(rows[-1]),flush=True)
    for p,h in sources.items():assert sha(p)==h
    audit={'passed':True,'all_counter_entries_exact':elements,'native_observer_updates':updates,'independent_raw_updates':updates,'aggregate_exceeds_uint16_demonstrated':True,'rows':rows,'source_sha256':sources,'scope':'All planned layer dimensions and both regimes bounded synthetic validation; no full classifier score or physical AND execution claimed.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in ['rows','source_sha256']},indent=2))
if __name__=='__main__':main()
