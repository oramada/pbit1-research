"""Official Torx one-site preparation versus flip-only stationary semantics."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,datetime,subprocess,math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from torx import psc
OUT=Path('results/thrml_spw/torx_probe/experiment')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrices(q,r):
    flip=np.array([[1-q,q],[q,1-q]],float);reset=np.array([[1,r],[0,1-r]],float);return flip@reset

def moments(M,initial,T):
    # Independent count moments conditional on current state; E[C^k 1(state=s)], k0..4.
    v=np.zeros((5,2));v[0,initial]=1
    for _ in range(T):
        nxt=np.zeros_like(v)
        for k in range(5):
            for b in [0,1]:
                for a in [0,1]:nxt[k,b]+=M[b,a]*sum(math.comb(k,l)*b**(k-l)*v[l,a] for l in range(k+1))
        v=nxt
    return np.array([v[k].sum()/T**k for k in range(5)])

@lru_cache(None)
def sample_kernel(T):
    sim=psc.BranchingSimulator(num_samples=1)
    def run(compiled,keys,initial):
        def chain(key):
            def step(state,k):
                nxt=sim.sample(compiled,state,k)[0];return nxt,nxt[0]
            _,path=jax.lax.scan(step,jnp.array([initial],jnp.int32),jax.random.split(key,T));return path
        return jax.vmap(chain)(keys)
    return eqx.filter_jit(run)

def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled;assert not (OUT/'protocol.json').exists();commit=subprocess.check_output(['git','-C','vendor/torx','rev-parse','HEAD'],text=True).strip();assert commit=='2142045276f24849a5ef2587a729ed961de8cf18'
    src={str(p):sha(p) for p in [Path(__file__),Path('vendor/torx/torx/psc/gates/_binary.py'),Path('vendor/torx/torx/psc/simulation/sampled.py'),Path('vendor/torx/torx/psc/simulation/statevector.py')]}
    modes=['flip_only','ideal_reset_then_flip','finite_reset_then_flip'];targets=[.1,.25,.5,.75,.9];Ts=[1,16,256]
    p={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'torx_commit':commit,'sources':src,'targets':targets,'modes':modes,'samples':Ts,'sampling_seeds':list(range(5)),'trials_per_seed':1000,'root_key':1467520,'mean_gate':'Fixed Bernstein bound on trial mean and its square, familyalpha.01 over90conditions*2 moments; includes2e-6 nominal F32 numerical allowance.','scope':'Single logical binary site and abstract Torx gates. Prepared distributions are observed only at whole-circuit boundaries; ideal reset is an explicit infinite-logit abstraction. No autonomous physical biased-cell, on-chip gate count or calibration/energy claim. Flip-only negative control is not the target stationary-occupancy representation.'};(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');sv=psc.StateVectorSimulator();bs=psc.BranchingSimulator(num_samples=1);rows=[];cycles=events=0;L=np.log(2*90*2/.01)
    for mi,mode in enumerate(modes):
        for pi,target in enumerate(targets):
            theta=jnp.asarray([np.log(target/(1-target))],jnp.float32);q=float(jax.nn.sigmoid(theta[0]));gates=[psc.PNOT(0)] if mi==0 else [psc.PReset(0),psc.PNOT(0)];params=[theta] if mi==0 else [jnp.asarray([jnp.inf if mi==1 else 8.],jnp.float32),theta];circuit=psc.DiscretePCircuit(gates);compiled=bs.build_circuit(circuit,params);dense=sv.build_circuit(circuit,params);assert compiled.num_pdits==1
            r=0. if mi==0 else 1. if mi==1 else float(jax.nn.sigmoid(jnp.float32(8)));M=matrices(q,r);measured=np.column_stack([np.asarray(sv.density(dense,jnp.asarray([1.,0.])),float),np.asarray(sv.density(dense,jnp.asarray([0.,1.])),float)]);np.testing.assert_allclose(measured,M,rtol=2e-6,atol=2e-7);stationary=q/(r+2*q*(1-r));eigen=(1-r)*(1-2*q)
            for initial in [0,1]:
                for T in Ts:
                    paths=[]
                    for seed in range(5):
                        key=jax.random.key(1467520)
                        for value in [mi,pi,initial,T,seed]:key=jax.random.fold_in(key,value)
                        paths.append(np.asarray(sample_kernel(T)(compiled,jax.random.split(key,1000),initial),np.uint8))
                    raw=np.stack(paths);assert np.all(raw<=1);emp=raw.mean(-1).reshape(-1);mom=moments(M,initial,T);v=mom[2]-mom[1]**2;v2=mom[4]-mom[2]**2;bounds=[np.sqrt(2*max(0,var)*L/5000)+L/(3*5000)+2e-6 for var in [v,v2]];obs=[float(emp.mean()),float((emp**2).mean())];ratios=[abs(obs[0]-mom[1])/bounds[0],abs(obs[1]-mom[2])/bounds[1]];passed=bool(max(ratios)<=1);name=f'mode{mi}_p{pi}_init{initial}_T{T}.npz';np.savez_compressed(OUT/name,packed_paths=np.packbits(raw,axis=-1,bitorder='little'),nominal_matrix=M,torx_density_matrix=measured,moments=mom);n=5000*T;cycles+=n;events+=len(gates)*n;row={'mode':mode,'target':target,'q_flip':q,'r_reset':r,'initial':initial,'T':T,'trials':5000,'stationary_one_probability':stationary,'lag1_eigenvalue':eigen,'expected_trial_mean':float(mom[1]),'observed_trial_mean':obs[0],'expected_trial_mean_variance':float(v),'observed_trial_mean_variance':float(emp.var(ddof=1)),'mean_and_second_moment_bound_ratios':ratios,'all_gates_passed':passed,'path':str(OUT/name),'path_sha256':sha(OUT/name),'logical_circuit_cycles':n,'logical_gate_applications':len(gates)*n};rows.append(row);print(json.dumps({k:row[k] for k in ['mode','target','initial','T','observed_trial_mean','all_gates_passed']}),flush=True)
    assert len(rows)==90 and cycles==40950000 and events==68250000
    for s,h in src.items():assert sha(s)==h
    result={'complete':True,'all_mean_moment_gates_passed':all(r['all_gates_passed'] for r in rows),'rows':rows,'logical_circuit_cycles':cycles,'logical_gate_applications':events,'scope':p['scope'],'new_neural_task_scores':False};(OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_mean_moment_gates_passed'];print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
