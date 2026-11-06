"""Official THRML trajectories for finite-controlled real GPT2 head weights."""
import os
os.environ['JAX_PLATFORMS']='cpu'
if hasattr(os,'sched_setaffinity'):
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-2:])
from pathlib import Path
import json,hashlib,time,math,importlib.metadata
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.native import NativeProbabilityBank

def main():
    root=OUT/'native_head_v1';protocol=json.loads((root/'protocol.json').read_text());assert sha(root/'inputs.npz')==protocol['inputs_sha256']
    assert sha(Path('research/thrml_spw/native.py'))==protocol['native_source_sha256']
    run={'source_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.json'),'thrml_version':importlib.metadata.version('thrml'),'numpy_version':np.__version__}
    with(root/'run.json').open('x')as f:json.dump(run,f,indent=2)
    begin=time.monotonic();data=np.load(root/'inputs.npz',allow_pickle=False);X=data['hidden'].astype(np.float64);W=data['original_weights'].astype(np.float64);results=[]
    for layout in protocol['layouts']:
        codes=data[layout+'_codes'];fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)
        requested=1/(1+np.exp(-2*fields.astype(np.float64)));bank=NativeProbabilityBank(requested)
        np.testing.assert_array_equal(bank.biases.reshape(fields.shape),fields)
        assert bank.logical_pbits==protocol['n_pbits'] and not bank.clamped_indices.size
        p=bank.realized_probabilities;lo=data[layout+'_lower'].astype(np.float64);span=data[layout+'_span'].astype(np.float64)
        mean=lo+span*p;var=span**2*p*(1-p);np.testing.assert_allclose(mean,data[layout+'_mean'],rtol=0,atol=1e-13)
        for seed in protocol['seeds']:
            states=bank.sample(seed,protocol['samples'],n_chains=1)[0];assert states.shape==(8192,8,768)
            packed=np.packbits(states,axis=0,bitorder='little');path=root/f'{layout}_seed{seed}_states.npz'
            with path.open('xb')as f:np.savez_compressed(f,packed_states=packed,shape=np.array(states.shape,np.int64))
            for T in protocol['nested_budgets']:
                counts=states[:T].sum(axis=0,dtype=np.int64);estimate=lo+span*counts/T;error=estimate-mean;v=var/T
                observed=float((error*error).sum());expected=float(v.sum());variance=float((2*v*v+span**4*p*(1-p)*(1-6*p*(1-p))/T**3).sum());z=(observed-expected)/math.sqrt(variance);assert abs(z)<8,(layout,seed,T,z)
                dot_error=X@error.T;dot_original=X@(estimate-W).T;dotvariance=X**2@v.T
                rec={'layout':layout,'seed':seed,'samples':T,'weight_SSE':observed,'predicted_weight_SSE':expected,'weight_SSE_z':z,
                    'dot_MSE_about_mean':float(np.mean(dot_error**2)),'predicted_dot_MSE_about_mean':float(dotvariance.mean()),'dot_MSE_about_original':float(np.mean(dot_original**2)),
                    'packed_states_sha256':sha(path),'native_probabilities_match_declared_fields':True}
                results.append(rec)
            (root/'progress.json').write_text(json.dumps({'layout':layout,'seed':seed,'elapsed_seconds':time.monotonic()-begin})+'\n')
            print(json.dumps({'layout':layout,'seed':seed,'elapsed_seconds':time.monotonic()-begin}),flush=True);del states,packed
    summary={}
    for T in protocol['nested_budgets']:
        summary[T]={}
        for layout in protocol['layouts']:
            r=[x for x in results if x['samples']==T and x['layout']==layout]
            assert len(r)==5
            summary[T][layout]={'measured_mean_dot_MSE':float(np.mean([x['dot_MSE_about_mean']for x in r])),'theoretical_mean_dot_MSE':r[0]['predicted_dot_MSE_about_mean']}
    result={'status':'completed_native_THRML_real_head_operator_validation','run':run,'records':results,'summary':summary,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']}
    with(root/'results.json').open('x')as f:json.dump(result,f,indent=2)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
