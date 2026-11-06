"""Independent finite-state audit of the existing correlated-dot evidence.

No producer imports, random sampling, or hardware assumptions. Rebuild Gibbs
blocks from Boltzmann conditional distributions, and finite-time variance by
Markov reward recursion rather than the producer's lag-covariance contraction.
"""
from pathlib import Path
import csv,hashlib,itertools,json,math
import numpy as np

ROOT=Path('results/thrml_spw/correlated_dot')
OUT=Path('results/thrml_spw/broad_verification_20261105/correlation_independent_audit_v1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def close(actual,expected):np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=3e-13)


def exact(n,fields,coupling):
    states=np.array(list(itertools.product([-1.,1.],repeat=n)))
    logmass=states@fields+coupling*np.sum(states[:,:-1]*states[:,1:],axis=1)
    pi=np.exp(logmass-logmass.max());pi/=pi.sum()
    transition=np.eye(len(states))
    for parity in (0,1):
        outside=np.arange(n)%2!=parity;block=np.zeros_like(transition)
        for i,state in enumerate(states):
            allowed=np.all(states[:,outside]==state[outside],axis=1)
            block[i,allowed]=pi[allowed]/pi[allowed].sum()
        close(block.sum(axis=1),1);close(pi@block,pi)
        transition=transition@block
    close(pi@transition,pi)
    return states,pi,transition


def reward_variance(pi,transition,values,steps):
    centered=values-pi@values
    first=pi*centered;second=pi*centered**2
    for _ in range(1,steps):
        transported=first@transition
        second=second@transition+2*centered*transported+pi*centered**2
        first=transported+pi*centered
    return float(second.sum()/steps**2)


def vectors(n):
    signs=np.where(np.arange(n)%2,-1.,1.)
    return {name:x/np.linalg.norm(x) for name,x in {
        'positive_uniform':np.ones(n),'mixed_alternating':signs,
        'mixed_graded':signs*np.arange(1,n+1)}.items()}


def audit(root=ROOT):
    names=['manifest.json','dot_product_mse.csv','warmup_stationarity.csv','raw_native_trajectories_and_dot_means.npz']
    inputs={name:sha(root/name) for name in names}
    with (root/'warmup_stationarity.csv').open()as f:models=list(csv.DictReader(f))
    with (root/'dot_product_mse.csv').open()as f:rows=list(csv.DictReader(f))
    assert len(models)==9 and len(rows)==108
    lookup={(int(r['n_spins']),float(r['coupling_J'])):r for r in models}
    assert set(lookup)==set(itertools.product([2,3,4],[0.,.35,.8]))
    used=set();ratios=[];errors=[];max_tv=0.;trajectory_bits=0
    with np.load(root/names[-1],allow_pickle=False)as raw:
        for (n,coupling),record in lookup.items():
            prefix=f'n{n}_J{coupling}';p=np.array([.2,.35,.65,.8])[:n]
            fields=np.array(json.loads(record['field_b']),np.float64)
            states,pi,P=exact(n,fields,coupling);z=(states+1)/2
            close(pi@z,p);close(json.loads(record['calibrated_target_p']),p)
            warmup=int(record['warmup_sweeps']);assert warmup==512
            initial=np.zeros(len(pi));initial[0]=1
            tv=float(np.abs(initial@np.linalg.matrix_power(P,warmup)-pi).sum()/2)
            close(tv,float(record['exact_tv_from_all_minus_initial']));assert tv<1e-10
            max_tv=max(max_tv,tv)
            centered=z-pi@z;propagated=centered.copy();cov=[]
            for lag in range(256):
                cov.append((centered.T*pi)@propagated);propagated=P@propagated
            cov=np.array(cov);close(cov,raw[prefix+'_exact_cov_z_by_lag'])
            spins=raw[prefix+'_spins'];assert spins.shape==(1000,256,n) and np.all((spins==-1)|(spins==1))
            trajectory_bits+=spins.size;draws=(spins.astype(np.float64)+1)/2
            for name,x in vectors(n).items():
                observable=z@x;mu=float(pi@observable)
                per_step=draws@x
                for T in [1,16,64,256]:
                    key=(n,coupling,name,T);used.add(key)
                    matches=[r for r in rows if (int(r['n_spins']),float(r['coupling_J']),r['input_vector'],int(r['samples_T']))==key]
                    assert len(matches)==1;r=matches[0]
                    assert int(r['trials'])==1000 and int(r['seed_groups'])==5 and int(r['trials_per_seed_group'])==200
                    close(json.loads(r['input_values']),x)
                    means=per_step[:,:T].mean(axis=1);close(means,raw[f'{prefix}_{name}_T{T}_means'])
                    differences=means-mu;squared=differences**2
                    mse=float(squared.mean());se=float(squared.std(ddof=1)/math.sqrt(1000))
                    expected=reward_variance(pi,P,observable,T)
                    naive=float(np.sum(x*x*p*(1-p))/T)
                    expected_fields={'target_dot_expectation':mu,'empirical_mean_of_trial_means':float(means.mean()),
                        'empirical_bias':float(differences.mean()),'empirical_mse':mse,'empirical_mse_se':se,
                        'empirical_mse_ci95_low':max(0,mse-1.96*se),'empirical_mse_ci95_high':mse+1.96*se,
                        'exact_full_covariance_mse':expected,'naive_independent_variance':naive,
                        'exact_to_naive_ratio':expected/naive,'empirical_to_exact_ratio':mse/expected,
                        'single_sample_spatial_variance':reward_variance(pi,P,observable,1),'warmup_exact_tv':tv}
                    for field,value in expected_fields.items():close(value,float(r[field]))
                    close(np.einsum('i,kij,j->k',x,cov[:T],x),json.loads(r['lag_covariances_lags_0_to_Tminus1']))
                    ratios.append(expected/naive);errors.append(abs(mse/expected-1))
    assert len(used)==108
    assert {name:sha(root/name) for name in names}==inputs
    return {'status':'all_existing_correlated_dot_tables_and_exact_finite_state_moments_independently_verified',
        'input_sha256':inputs,'source_sha256':sha(__file__),'models':9,'conditions':108,
        'saved_native_state_bits_checked':trajectory_bits,'exact_to_naive_ratio_min':min(ratios),
        'exact_to_naive_ratio_max':max(ratios),'mean_abs_empirical_to_exact_ratio_error':float(np.mean(errors)),
        'max_warmup_total_variation':max_tv,
        'scope':'Rebuilds exact stationary Gibbs kernels from saved fields, all lag matrices, Markov-reward finite-T moments and every reported empirical metric from saved raw trajectories. Does not independently replay native random bits, prove large-model robustness, or characterize physical device costs.'}


def main():
    result=audit();OUT.mkdir(exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
