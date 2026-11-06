"""Independent Torx path replay and full count-distribution Markov recursion."""
from pathlib import Path
from functools import lru_cache
import json,hashlib
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from torx import psc
OUT=Path('results/thrml_spw/torx_probe/experiment')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(None)
def replay_kernel(T):
    sim=psc.BranchingSimulator(num_samples=1)
    def run(compiled,keys,initial):
        def one(k):
            time_keys=jax.random.split(k,T)
            def step(i,carry):
                state,buffer=carry;nxt=sim.sample(compiled,state,time_keys[i])[0];return nxt,buffer.at[i].set(nxt[0])
            return jax.lax.fori_loop(0,T,step,(jnp.asarray([initial],jnp.int32),jnp.zeros(T,jnp.int32)))[1]
        return jax.vmap(one)(keys)
    return eqx.filter_jit(run)
def count_distribution(M,initial,T):
    mass=np.zeros((2,T+1));mass[initial,0]=1
    for _ in range(T):
        nxt=np.zeros_like(mass);nxt[0]=M[0,0]*mass[0]+M[0,1]*mass[1];nxt[1,1:]=M[1,0]*mass[0,:-1]+M[1,1]*mass[1,:-1];mass=nxt
    np.testing.assert_allclose(mass.sum(),1,rtol=1e-12,atol=1e-12);u=np.arange(T+1)/T;return np.array([np.dot(mass.sum(0),u**k) for k in range(5)])
def main():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and m['all_mean_moment_gates_passed'];assert not (OUT/'evidence_audit.json').exists()
    for s,h in p['sources'].items():assert sha(s)==h
    bs=psc.BranchingSimulator(num_samples=1);bits=cycles=events=0;L=np.log(2*90*2/.01)
    for row in m['rows']:
        mode=row['mode'];mi=p['modes'].index(mode);pi=p['targets'].index(row['target']);initial,T=row['initial'],row['T'];assert sha(row['path'])==row['path_sha256'];a=np.load(row['path']);stored=np.unpackbits(a['packed_paths'],axis=-1,bitorder='little')[:,:,:T];assert stored.shape==(5,1000,T)
        if T%8:assert not np.unpackbits(a['packed_paths'],axis=-1,bitorder='little')[:,:,T:].any()
        gates=[psc.PNOT(0)] if mi==0 else [psc.PReset(0),psc.PNOT(0)];theta=jnp.array([np.log(row['target']/(1-row['target']))],jnp.float32);params=[theta] if mi==0 else [jnp.array([jnp.inf if mi==1 else 8.],jnp.float32),theta];compiled=bs.build_circuit(psc.DiscretePCircuit(gates),params);assert compiled.num_pdits==1
        for seed in range(5):
            key=jax.random.key(1467520)
            for v in [mi,pi,initial,T,seed]:key=jax.random.fold_in(key,v)
            replay=np.asarray(replay_kernel(T)(compiled,jax.random.split(key,1000),initial),np.uint8);np.testing.assert_array_equal(replay,stored[seed])
        q,r=row['q_flip'],row['r_reset'];M=np.array([[1-q,(1-q)*r+q*(1-r)],[q,q*r+(1-q)*(1-r)]],float);np.testing.assert_allclose(M,a['nominal_matrix'],rtol=1e-14,atol=1e-14);np.testing.assert_allclose(M,a['torx_density_matrix'],rtol=2e-6,atol=2e-7);mom=count_distribution(M,initial,T);np.testing.assert_allclose(mom,a['moments'],rtol=2e-12,atol=2e-12)
        stationary=M[1,0]/(M[1,0]+M[0,1]);eigen=1-M[1,0]-M[0,1];np.testing.assert_allclose([stationary,eigen],[row['stationary_one_probability'],row['lag1_eigenvalue']],rtol=1e-12,atol=1e-12)
        empirical=stored.astype(float).mean(-1).reshape(-1);var=mom[2]-mom[1]**2;var2=mom[4]-mom[2]**2;bound=np.array([np.sqrt(2*max(0,v)*L/5000)+L/(3*5000)+2e-6 for v in [var,var2]]);ratios=np.abs(np.array([empirical.mean(),(empirical**2).mean()])-mom[1:3])/bound
        np.testing.assert_allclose(ratios,row['mean_and_second_moment_bound_ratios'],rtol=1e-8,atol=1e-8);assert np.all(ratios<=1);np.testing.assert_allclose([empirical.mean(),empirical.var(ddof=1),var],[row['observed_trial_mean'],row['observed_trial_mean_variance'],row['expected_trial_mean_variance']],rtol=1e-10,atol=1e-12)
        bits+=stored.size;cycles+=stored.size;events+=stored.size*len(gates);print(f'Replayed Torx {mode} p{row["target"]} init{initial} T{T}',flush=True)
    assert bits==cycles==40950000 and events==68250000;result={'passed':True,'conditions_checked':90,'raw_bit_outputs_reproduced_exactly':bits,'independent_replay_logical_cycles':cycles,'independent_replay_logical_gate_applications':events,'all_density_matrices_and_count_moments_checked':True,'all_180_fixed_mean_moment_gates_passed':True,'manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'scope':'Separate public Torx sampling loop and independent count-distribution recursion. Same-key verification, not additional independent MC replicates. One logical site at full circuit boundaries; no physical calibration or device-count inference.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
