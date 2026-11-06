"""Independently decode saved THRML states and replay one chain per layout."""
import os
os.environ['JAX_PLATFORMS']='cpu'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
import json,hashlib,math
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.native import NativeProbabilityBank

def main():
    root=OUT/'native_head_v1';report=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text());dest=OUT/'native_head_audit_v1';dest.mkdir(exist_ok=False)
    assert len(report['records'])==40
    arrays=np.load(root/'inputs.npz',allow_pickle=False);X=arrays['hidden'].astype(np.float64);checked=0;replayed=[]
    for layout in protocol['layouts']:
        f=arrays[layout+'_codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4);p=1/(1+np.exp(-2*f.astype(np.float64)))
        low=arrays[layout+'_lower'].astype(np.float64);span=arrays[layout+'_span'].astype(np.float64);mu=low+span*p
        for seed in protocol['seeds']:
            file=root/f'{layout}_seed{seed}_states.npz'
            with np.load(file,allow_pickle=False)as d:states=np.unpackbits(d['packed_states'],axis=0,count=8192,bitorder='little').astype(bool);assert tuple(d['shape'])==states.shape==(8192,8,768)
            if seed==protocol['seeds'][0]:
                native=NativeProbabilityBank(p).sample(seed,8192,1)[0];assert np.array_equal(native,states);replayed.append({'layout':layout,'seed':seed,'states':int(states.size)})
            for T in protocol['nested_budgets']:
                r=next(r for r in report['records']if r['layout']==layout and r['seed']==seed and r['samples']==T);assert sha(file)==r['packed_states_sha256']
                e=low+span*np.count_nonzero(states[:T],axis=0)/T-mu;v=span**2*p*(1-p)/T;sse=float(np.sum(e**2));theory=float(v.sum())
                var=float(np.sum(2*v**2+span**4*p*(1-p)*(1-6*p*(1-p))/T**3));z=(sse-theory)/math.sqrt(var)
                measured=float(np.mean((X@e.T)**2));pred=float(np.mean(X**2@v.T))
                assert np.isclose(sse,r['weight_SSE'],atol=1e-12,rtol=1e-10)
                assert np.isclose(measured,r['dot_MSE_about_mean'],atol=1e-12,rtol=1e-10)
                assert np.isclose(pred,r['predicted_dot_MSE_about_mean'],atol=1e-12,rtol=1e-10)
                assert abs(z)<8;checked+=1
    result={'status':'passed_40_saved_trajectory_checks_and_two_exact_native_replays','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),
        'checked_conditions':checked,'replayed':replayed,'scope':protocol['scope']}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for layout,color,label in [('storage_flat','tomato','Original grouping'),('transposed_matrix','teal','Reoriented grouping')]:
        T=np.array(protocol['nested_budgets']);measured=[];theory=[]
        for t in T:
            rows=[r for r in report['records']if r['layout']==layout and r['samples']==t];measured.append(np.mean([r['dot_MSE_about_mean']for r in rows]));theory.append(rows[0]['predicted_dot_MSE_about_mean'])
        ax.loglog(T,theory,'--',color=color,label=label+' theory');ax.loglog(T,measured,'o-',color=color,label=label+' native THRML')
    ax.set(xlabel='Samples per p-bit',ylabel='Output MSE about finite-control mean',title='Real GPT-2 output weights: native THRML operator audit\n6,144 p-bits, five seeds, eight fixed hidden vectors');ax.legend();fig.tight_layout();fig.savefig(dest/'native_head_variance.png',dpi=160);fig.savefig(dest/'native_head_variance.svg')
    print(json.dumps(result));print(json.dumps(report['summary']['8192'],indent=2))

if __name__=='__main__':main()
