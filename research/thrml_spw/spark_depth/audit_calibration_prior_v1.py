"""Independent optimizer check and calibration-cost visualization."""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .prepare_v1 import OUT,sha
from .calibration_prior_v1 import fit

def main():
    root=OUT/'calibration_prior_v1';dest=OUT/'calibration_prior_audit_v1';dest.mkdir(exist_ok=False);result=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    assert sha(root/'probe_counts.npz')==result['probe_counts_sha256'];compared=0;maximum_difference=0.
    with np.load(root/'probe_counts.npz',allow_pickle=False)as f:
        x=f['probe_logit'];D=np.column_stack([x,np.ones(3)])
        for layout in ['storage_flat','transposed_matrix']:
            for N in [256,4096,65536]:
                counts=f[f'{layout}_2026111101_{N}'];assert counts.shape==(8,768,3)and np.all(counts<=N)
                for prior in [False,True]:
                    ah,bh=fit(counts,N,x,prior);precision=2500. if prior else 0.
                    for head,column in [(0,0),(0,496),(7,430),(7,767)]:
                        y=counts[head,column].astype(float)
                        def fun(t):
                            eta=D@t;return np.sum(N*np.logaddexp(0,eta)-y*eta)+.5*precision*((t[0]-1)**2+t[1]**2)
                        def jac(t):return D.T@(N*expit(D@t)-y)+precision*(t-np.array([1.,0.]))
                        def hess(t):
                            q=expit(D@t);return D.T@((N*q*(1-q))[:,None]*D)+precision*np.eye(2)
                        optimum=minimize(fun,[1.,0.],method='trust-exact',jac=jac,hess=hess,options={'gtol':1e-8})
                        delta=np.max(np.abs(optimum.x-[ah[head,column],bh[head,column]]));assert delta<2e-6,delta
                        compared+=1;maximum_difference=max(maximum_difference,float(delta))
    for row in result['records']:
        assert np.isfinite(row['expected_output_MSE'])and row['expected_output_MSE']>=row['mean_squared_bias']>=0
        if row['method']!='uncalibrated':assert row['calibration_shots']==3*row['probes_per_operating_point']*6144
    audit={'status':'independent_binomial_optimizer_and_probe_cost_checks_passed','source_sha256':sha(__file__),'results_sha256':sha(root/'results.json'),'independent_optimizer_comparisons':compared,'maximum_coefficient_difference':maximum_difference,
        'scope':'ConditionaloperatorMSEsimulation. Correctpopulationpriorisassumed,notmeasured. Optimizeragreementdoesnotprovephysicalcalibrationorwholemodelquality.'}
    (dest/'results.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,layout,label in zip(axs,['storage_flat','transposed_matrix'],['Original grouping','Reoriented grouping']):
        base=np.mean([r['expected_output_MSE']for r in result['records']if r['layout']==layout and r['method']=='uncalibrated'])
        ax.axhline(base,color='gray',linestyle='--',label='Uncalibrated')
        for method,color in [('MLE','tomato'),('MAP','teal')]:
            values=[np.mean([r['expected_output_MSE']for r in result['records']if r['layout']==layout and r['method']==method and r['probes_per_operating_point']==N])for N in [256,4096,65536]]
            ax.loglog(np.array([256,4096,65536])*3/32768,values,'o-',color=color,label=method+(' (assumed prior)'if method=='MAP'else''))
        ax.set(xlabel='Calibration probes / one T=32,768 reconstruction',ylabel='Expected output MSE',title=label);ax.legend(fontsize=8)
    fig.suptitle('Illustrative device mismatch: calibration has a sampling cost');fig.tight_layout();fig.savefig(dest/'calibration_cost.png',dpi=160);fig.savefig(dest/'calibration_cost.svg')
    print(json.dumps(audit))

if __name__=='__main__':main()
