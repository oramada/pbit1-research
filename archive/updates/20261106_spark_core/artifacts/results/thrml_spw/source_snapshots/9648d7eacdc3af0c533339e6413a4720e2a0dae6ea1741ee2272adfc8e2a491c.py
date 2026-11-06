"""Native consecutive-state activity witness and full learned-control ledger."""
import json,os,subprocess
from pathlib import Path
from functools import lru_cache
from importlib.metadata import version
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state
from .native import NativeProbabilityBank
from .controlled_positive_native import set_fields
from .state_activity_math import OUT,sha,moments,full_ledger
class Activity(AbstractObserver):
    readout:object
    transitions:int
    def init(self):return (jnp.zeros(256,jnp.bool_),jnp.zeros(256,jnp.bool_),jnp.zeros(256,jnp.bool_))+tuple(jnp.zeros(256,jnp.int32) for _ in range(6))
    def __call__(self,program,free,clamped,carry,iteration):
        global_state=block_state_to_global(free+clamped,program.gibbs_spec);z=from_global_state(global_state,program.gibbs_spec,[self.readout])[0];previous,first,reset_last,*counts=carry;delta=(z!=previous).astype(jnp.int32);up=((~previous)&z).astype(jnp.int32);down=(previous&(~z)).astype(jnp.int32);reset=iteration<self.transitions;retained=iteration>0;flags=[reset,reset,reset,retained,retained,retained];increments=[delta,up,down,delta,up,down];new=tuple(c+jnp.where(flag,inc,0) for c,flag,inc in zip(counts,flags,increments));return (z,jnp.where(iteration==0,z,first),jnp.where(iteration==self.transitions-1,z,reset_last))+new,None

def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled;assert not (OUT/'manifest.json').exists();OUT.mkdir(parents=True,exist_ok=True);assert __import__('shutil').disk_usage(OUT).free>100000000;parent=Path('results/thrml_spw/positional_student');closed=json.loads((parent/'root_review.json').read_text());assert closed['phase']=='completed_and_independently_audited_goal_active';old=json.loads((parent/'protocol.json').read_text());sources={str(Path('research/thrml_spw')/name):sha(Path('research/thrml_spw')/name) for name in ['state_activity.py','state_activity_math.py','verify_state_activity.py','state_activity_report.py','test_state_activity_math.py','native.py','controlled_positive_native.py']};commit=subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip();assert commit=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for entry in old['matrices']:
        for key in ['single_controls','positional_controls']:assert sha(entry[key]['path'])==entry[key]['sha256']
    protocol={'official_commit':commit,'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy']},'parent_root_review_sha256':sha(parent/'root_review.json'),'parent_protocol_sha256':sha(parent/'protocol.json'),'matrices':old['matrices'],'source_sha256':sources,'controller_codes':256,'field_bound':4,'transition_budgets':[1,4,16,64,256],'seeds':list(range(5)),'trials_per_seed':200,'chain_batch':50,'key_root':1287520,'witness_sampling':'One official call emits T+1 states. FirstT records count changes fromFalse; allT gaps between successive records count retained/stationary changes. Charge allT+1 emitted updates. This is a separate witness, not byte replay of previous T-sample neural trajectories.','gate':'Public state replay reproduces every counter exactly; count direction/telescoping identities exact. Nominal IID mean changes within familyalpha.01 Bernstein bounds across2*5*256 conditions, using independently verified means/variances. No physical energy/quality gate.','scope':'Post-hoc activity proxy for frozen learned controls and a native256-code witness. No new neural scores, model selection or hardware switching/energy measurement. Physical changes between sample instants, holding power, programming, idle and correlations remain unmeasured.','XLA_FLAGS':os.environ.get('XLA_FLAGS')};(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows,totals=full_ledger(old['matrices']);(OUT/'ledger.json').write_text(json.dumps({'rows':rows,'totals':totals},indent=2)+'\n');fields=(-4+np.arange(256,dtype=float)*8/255).astype(np.float32);p=np.exp(-np.logaddexp(0,-2*fields.astype(float)));bank=NativeProbabilityBank(np.full(256,.5));initial=[jnp.zeros(256,jnp.bool_)];cases=[];updates=0;names=['last','first','reset_last','reset_flips','reset_up','reset_down','retained_flips','retained_up','retained_down']
    @lru_cache(None)
    def kernel(t):
        def batch(keys,fields):
            program=set_fields(bank,fields)
            def one(key):
                obs=Activity(bank.readout,t);out,_=sample_with_observation(key,program,SamplingSchedule(1,t+1,1),initial,[],obs.init(),obs);return out
            return jax.vmap(one)(keys)
        return jax.jit(batch)
    for t in protocol['transition_budgets']:
        arrays={name:np.empty((5,200,256),np.bool_ if name in names[:3] else np.int32) for name in names}
        for si,seed in enumerate(protocol['seeds']):
            for first in range(0,200,50):
                key=jax.random.key(protocol['key_root'])
                for value in [seed,t,first]:key=jax.random.fold_in(key,value)
                output=kernel(t)(jax.random.split(key,50),jnp.asarray(fields))
                for name,value in zip(names,output):arrays[name][si,first:first+50]=np.asarray(value)
                updates+=50*(t+1)*256
        np.testing.assert_array_equal(arrays['reset_up']+arrays['reset_down'],arrays['reset_flips']);np.testing.assert_array_equal(arrays['retained_up']+arrays['retained_down'],arrays['retained_flips']);np.testing.assert_array_equal(arrays['reset_up']-arrays['reset_down'],arrays['reset_last']);np.testing.assert_array_equal(arrays['retained_up']-arrays['retained_down'],arrays['last'].astype(int)-arrays['first'].astype(int));path=OUT/f'T{t}.npz';np.savez_compressed(path,fields=fields,probabilities=p,**arrays);checks={}
        for mode,stationary in [('reset',False),('retained',True)]:
            mean,var=moments(p,t,stationary);observed=arrays[mode+'_flips'].mean((0,1));L=np.log(2*2*5*256/.01);bound=np.sqrt(2*var*L/1000)+t*L/3000;checks[mode]={'all_means_within_Bernstein':bool(np.all(np.abs(observed-mean)<=bound)),'max_mean_error_over_bound':float(np.max(np.abs(observed-mean)/bound))}
        cases.append({'T':t,'path':str(path),'sha256':sha(path),'checks':checks,'native_node_updates':1000*(t+1)*256});print(f'Native reset/retained activity T{t}: both counter families complete',flush=True)
    for source,h in sources.items():assert sha(source)==h
    result={'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'ledger_sha256':sha(OUT/'ledger.json'),'cases':cases,'native_node_updates':updates,'all_mean_gates_passed':all(v['all_means_within_Bernstein'] for c in cases for v in c['checks'].values()),'new_task_scores':0,'scope':protocol['scope']};(OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['all_mean_gates_passed']
if __name__=='__main__':main()
