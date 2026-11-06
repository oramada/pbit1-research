"""All-example replay via public THRML weight frequencies, independent of NN observer."""
from functools import lru_cache
import csv
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import (AbstractObserver,Block,BlockGibbsSpec,SpinNode,FactorSamplingProgram,
    SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation)
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/calibrated_mnist');DATA=Path('data/thrml_spw/positive_mnist')


class FrequencyObserver(AbstractObserver):
    readout:object
    size:int
    def init(self):return jnp.zeros(self.size,dtype=jnp.int32)
    def __call__(self,program,state_free,state_clamped,carry,iteration):
        state=block_state_to_global(state_free+state_clamped,program.gibbs_spec)
        bits=from_global_state(state,program.gibbs_spec,[self.readout])[0]
        return carry+bits.astype(jnp.int32),None


class PublicLayer:
    def __init__(self,weight):
        flat=weight.reshape(-1);self.shape=weight.shape;self.size=flat.size
        free=np.flatnonzero((flat>0)&(flat<1));clamped=np.flatnonzero((flat==0)|(flat==1))
        nodes=[SpinNode() for _ in flat];self.readout=Block(nodes)
        self.free=[Block([nodes[i] for i in free])];self.clamped=[Block([nodes[i] for i in clamped])] if len(clamped) else []
        self.spec=BlockGibbsSpec(self.free,self.clamped,{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)})
        self.clamp_state=[jnp.asarray(flat[clamped],dtype=jnp.bool_)] if len(clamped) else []
        self.initial=[jnp.zeros(len(free),dtype=jnp.bool_)];self.n_free=len(free)

    @lru_cache(maxsize=4)
    def frequencies(self,t):
        def run(keys,fields):
            # Public factor construction, rather than the production helper that
            # replaces an already compiled interaction's unary array.
            program=FactorSamplingProgram(self.spec,[SpinGibbsConditional()],
                [SpinEBMFactor(self.free,fields)],[])
            def one(key):
                observer=FrequencyObserver(self.readout,self.size)
                total,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),
                    self.initial,self.clamp_state,observer.init(),observer)
                return (total.astype(jnp.float32)/t).reshape(self.shape)
            return jax.vmap(one)(keys)
        return jax.jit(run)


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Replay must use original CUDA backend')
    p=json.loads((OUT/'protocol.json').read_text());assert sha(DATA/'evaluation_data.npz')==p['input_sha256']
    for path,h in p['source_sha256'].items():assert sha(path)==h
    d=np.load(DATA/'evaluation_data.npz');x=d['test_inputs'];xb=np.pad(x,((0,16),(0,0))).reshape(313,32,64)
    with (OUT/'native_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==750;errors=[]
    for model in range(5):
        ex=np.load(DATA/f'seed{model}.npz');w=[ex[f'weight_{i}'] for i in range(2)];bias=[ex[f'bias_{i}'] for i in range(2)]
        layers=[PublicLayer(v) for v in w];split=layers[0].n_free
        fn={(i,t):layers[i].frequencies(t) for i in [0,1] for t in [64,256,1024]}
        for device in range(10):
            bundle=np.load(OUT/'devices'/f'model{model}_device{device}.npz')
            for t in [64,256,1024]:
                key=jax.random.key(1080730)
                for v in [model,device,t]:key=jax.random.fold_in(key,v)
                batches=jax.vmap(lambda i:jax.random.fold_in(key,i))(jnp.arange(313))
                kk=jax.vmap(lambda k:jax.random.split(k,2))(batches);k0,k1=kk[:,0],kk[:,1]
                for condition in p['conditions']:
                    fields=bundle[f'fields_{condition}'];name=f'model{model}_device{device}_{condition}_T{t}'
                    raw=np.load(OUT/'runs'/f'{name}.npz');saved_pre,saved_logits=raw['preactivation0'],raw['logits']
                    q0=np.asarray(fn[0,t](k0,jnp.asarray(fields[:split])))
                    expected_pre=(np.einsum('bni,bio->bno',xb.astype(float),q0.astype(float))+bias[0]).reshape(-1,32)[:10000]
                    pre_error=float(np.max(np.abs(expected_pre-saved_pre))/(1+np.max(np.abs(expected_pre))))
                    assert pre_error<2e-5,(name,'first preactivation',pre_error)
                    hidden=np.maximum(saved_pre,0);hb=np.pad(hidden,((0,16),(0,0))).reshape(313,32,32)
                    q1=np.asarray(fn[1,t](k1,jnp.asarray(fields[split:])))
                    expected_logits=(np.einsum('bni,bio->bno',hb.astype(float),q1.astype(float))+bias[1]).reshape(-1,10)[:10000]
                    logit_error=float(np.max(np.abs(expected_logits-saved_logits))/(1+np.max(np.abs(expected_logits))))
                    assert logit_error<2e-5,(name,'second preactivation',logit_error)
                    errors.append({'model':model,'device':device,'condition':condition,'T':t,'scaled_max_first_preactivation_error':pre_error,'scaled_max_logit_error':logit_error})
            print(f'Public frequency reconstruction verified every example, model{model} device{device}',flush=True)
        jax.clear_caches()
    assert len(errors)==750
    with (OUT/'public_path_errors.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(errors[0]));wr.writeheader();wr.writerows(errors)
    audit={'passed':True,'native_runs_audited':750,'examples_per_run':10000,'layers_per_run':2,'batches_per_run':313,
        'max_scaled_first_preactivation_error':max(r['scaled_max_first_preactivation_error'] for r in errors),
        'max_scaled_logit_error':max(r['scaled_max_logit_error'] for r in errors),'tolerance':2e-5,'source_sha256':sha(__file__),
        'scope':'All full-test examples replayed from separately constructed public THRML factors and exact integer state frequencies, independent of production output-sum observer. F64 matrix reconstruction checks saved layer0 means, ReLU and layer1 logits to F32 summation tolerance. Same CUDA backend and field/key schedule; no physical claim.'}
    (OUT/'public_path_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
