"""Recompute confirmation statistics from token-loss arrays and enforce cohort completeness."""
import json,math,hashlib
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'confirm_layout_v1';result=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text());dest=OUT/'confirmation_audit_v1';dest.mkdir(exist_ok=False)
    assert sha(root/'protocol.json')==result['protocol_sha256']
    for file,key in [('confirm_layout_v1.py','source_sha256'),('causal_v1.py','sampler_sha256'),('trace_v1.py','metric_sha256')]:assert sha(__import__('pathlib').Path(__file__).with_name(file))==protocol[key]
    conditions=['BF16']+[f'{layout}_seed{seed}'for seed in protocol['seeds']for layout in protocol['layouts']]
    assert len(result['records'])==len(conditions)*24
    values={};with_point={}
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as f:ids=f['input_ids'][:,:257].copy()
    for name in conditions:
        rows=[r for r in result['records']if r['condition']==name];assert sorted(r['row']for r in rows)==list(range(8,32))
        assert len({r['point_sha256']for r in rows})==1;with_point[name]=rows[0]['point_sha256'];values[name]={}
        for r in rows:
            row=r['row'];loss=np.load(root/f'{name}_row{row}_NLL.npy',allow_pickle=False);assert loss.shape==(256,)and np.isfinite(loss).all();assert abs(loss.mean()-r['metrics']['mean_NLL'])<1e-12
            assert hashlib.sha256(ids[row].tobytes()).hexdigest()==r['input_sha256'];values[name][row]=float(loss.mean())
    recomputed={}
    for split,start,stop in [('calibration',8,16),('confirmation',16,32)]:
        base=np.mean([values['BF16'][r]for r in range(start,stop)]);recomputed[split]={'BF16_PPL':float(np.exp(base)),'layouts':{}}
        for layout in protocol['layouts']:
            losses=np.array([np.mean([values[f'{layout}_seed{seed}'][r]for r in range(start,stop)])for seed in protocol['seeds']]);delta=losses-base;half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5);interval=np.exp([delta.mean()-half,delta.mean()+half]);meanratio=np.exp(delta.mean());passed=bool(delta.mean()+half<=np.log(1.02))
            recorded=result['summaries'][split]['layouts'][layout];assert np.allclose(interval,recorded['MC95_t_PPL_ratio_interval'],rtol=0,atol=1e-12);assert passed==recorded['frozen_2pct_gate_pass'];assert abs(meanratio-recorded['mean_PPL_ratio'])<1e-12
            recomputed[split]['layouts'][layout]={'mean_PPL_ratio':float(meanratio),'MC95_t_PPL_ratio_interval':interval.tolist(),'passes_frozen_2pct_gate':passed}
    ledgers={}
    for name in conditions[1:]:
        a=json.loads((root/f'{name}_sampling.json').read_text());p=a['parameters'];assert len(p)==148
        assert sum(x['parameters']for x in p)==124439808;assert a['total_Bernoulli_observations']==124439808*8192
        var=sum(x['variance_of_SSE']for x in p);z=(sum(x['SSE_about_finite_control_mean_before_FP32_BF16_rounding']for x in p)-sum(x['expected_SSE']for x in p))/math.sqrt(var);assert abs(z)<8
        assert a['range_metadata_bits']==64*sum(x['range_groups']for x in p)
        ledgers[name]={'moment_z':z,'logical_pbits':124439808,'observations':a['total_Bernoulli_observations'],'ideal_control_range_state_bits_per_weight':13+a['range_metadata_bits']/124439808}
    audit={'status':'passed_independent_recomputation_and_complete_five_seed_cohorts','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'records':len(result['records']),'targets_per_confirmation_seed':4096,'targets_per_calibration_seed':2048,'recomputed':recomputed,'ledgers':ledgers,
        'scope':'Exactsavedtokenlossrecomputation,notindependentforwardimplementation. CIsmeasureMCvariationonfixedreservedtrainingarticles,notbroadgeneralization.'}
    (dest/'results.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5));labels=['Original grouping','Reoriented grouping'];c=recomputed['confirmation']['layouts']
    for i,layout in enumerate(protocol['layouts']):
        d=c[layout];mean=d['mean_PPL_ratio'];lo,hi=d['MC95_t_PPL_ratio_interval'];ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=7,color=['tomato','teal'][i])
    ax.axhline(1,color='gray',label='BF16');ax.axhline(1.02,color='orange',linestyle='--',label='Frozen 2% tolerance');ax.set_xticks([0,1],labels);ax.set(ylabel='Perplexity / BF16 perplexity',title='GPT-2: five fresh seeds, 4,096 reserved-training targets\nOne p-bit per parameter, 8,192 samples');ax.legend();fig.tight_layout();fig.savefig(dest/'five_seed_confirmation.png',dpi=160);fig.savefig(dest/'five_seed_confirmation.svg')
    print(json.dumps(recomputed,indent=2))

if __name__=='__main__':main()
