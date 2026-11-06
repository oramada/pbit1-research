"""Audit causal outputs and explain grouping through fixed-hidden logit variance."""
from pathlib import Path
import json,hashlib,math,csv
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'causal_v1';raw=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text());dest=OUT/'explanation_v1';dest.mkdir(exist_ok=False)
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as f:ids=f['input_ids'][:8,:129].copy()
    trace=json.loads((OUT/'trace_v1/results.json').read_text());tracebase={r['row']:r for r in trace['records']if r['point']=='original_BF16'}
    assert len(raw['records'])==len(protocol['conditions'])*8
    conditions={}
    for condition in protocol['conditions']:
        name=condition['name'];rows=[r for r in raw['records']if r['condition']==name];assert sorted(r['row']for r in rows)==list(range(8))
        values=[]
        for r in rows:
            nll=np.load(root/f"{name}_row{r['row']}_NLL.npy",allow_pickle=False);assert nll.shape==(128,)and np.isfinite(nll).all()
            assert abs(nll.mean()-r['metrics']['mean_NLL'])<1e-12
            assert hashlib.sha256(ids[r['row']].tobytes()).hexdigest()==r['input_sha256']
            if name=='original_BF16':assert r['logits_sha256']==tracebase[r['row']]['logits_sha256']
            values.extend(nll)
        assert len(set(r['point_sha256']for r in rows))==1
        conditions[name]={'mean_NLL':float(np.mean(values)),'perplexity':math.exp(float(np.mean(values))),
            'restored_BF16_parameters':rows[0]['BF16_restored_parameters'],'point_sha256':rows[0]['point_sha256']}
    audits={}
    for name,draw in raw['draw_reports'].items():
        params=draw['parameters'];assert len(params)==148 and sum(r['parameters']for r in params)==124439808
        expected=sum(r['expected_SSE']for r in params);sse=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']for r in params);var=sum(r['variance_of_SSE']for r in params)
        z=(sse-expected)/math.sqrt(var);assert abs(z)<8 and abs(z-draw['aggregate_SSE_z'])<1e-12
        assert draw['total_Bernoulli_observations']==124439808*8192
        assert draw['controller_bits']==12*124439808
        assert draw['range_metadata_bits']==64*sum(r['range_groups']for r in params)
        audits[name]={'moment_z':z,'ideal_bits_per_parameter_including_control_ranges_and_cell':13+draw['range_metadata_bits']/124439808}
    hidden=[]
    for i in range(8):
        with np.load(OUT/'trace_v1/original_BF16'/f'row{i}.npz',allow_pickle=False)as f:hidden.append(f['transformer.ln_f'][:,:-1].reshape(-1,768).astype(np.float64))
    mean_squared_hidden=(np.concatenate(hidden)**2).mean(0);noise={}
    with np.load(OUT/'weight_survey_v1/axis_statistics.npz',allow_pickle=False)as f:
        for layout in ['storage_flat','transposed_matrix']:
            column_var=f[f'p000_{layout}_axis1_noise_variance_coefficient']/50257/8192
            contribution=mean_squared_hidden*column_var;order=np.argsort(contribution)[::-1]
            noise[layout]={'mean_conditional_logit_variance':float(contribution.sum()),
                'top10_channels':[{'channel':int(i),'contribution':float(contribution[i]),'mean_squared_hidden':float(mean_squared_hidden[i]),'mean_weight_variance_T8192':float(column_var[i])}for i in order[:10]]}
    report={'status':'passed_saved_array_metrics_input_hash_baseline_replay_moment_and_ledger_audits','source_sha256':sha(__file__),
        'causal_results_sha256':sha(root/'results.json'),'trace_results_sha256':sha(OUT/'trace_v1/results.json'),
        'weight_survey_sha256':sha(OUT/'weight_survey_v1/results.json'),'audited_forward_records':len(raw['records']),
        'conditions':conditions,'sampling_ledgers':audits,'fixed_hidden_logit_variance':noise,
        'fixed_hidden_variance_reduction_factor':noise['storage_flat']['mean_conditional_logit_variance']/noise['transposed_matrix']['mean_conditional_logit_variance'],
        'scope':'Descriptive single-seed trainingdiagnostics. Fixed-hidden analytic output variance excludes changes in hidden activations and cross-token dependence. No universal, physicalhardware, or directexecutionclaim.'}
    (dest/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    points=['original_BF16','ideal_counts_flat_G128','restore_token_embedding','ideal_counts_transposed_G128','ideal_counts_transposed_G32'];labels=['BF16','Original grouping','Embedding/head restored\n(diagnostic only)','Reoriented G128','Reoriented G32']
    ax[0].barh(labels,[conditions[p]['perplexity']for p in points],color=['gray','tomato','slategray','teal','steelblue']);ax[0].set(xlabel='Perplexity; lower is better',title='One exploratory seed, 1,024 training targets');ax[0].invert_yaxis()
    channels=[496,430,36,314];flat={x['channel']:x['contribution']for x in noise['storage_flat']['top10_channels']};trans={x['channel']:x['contribution']for x in noise['transposed_matrix']['top10_channels']};x=np.arange(len(channels))
    # Full four-channel values computed from original arrays, not truncated top10.
    with np.load(OUT/'weight_survey_v1/axis_statistics.npz',allow_pickle=False)as f:
        for offset,layout,color,label in [(-.18,'storage_flat','tomato','Original grouping'),(.18,'transposed_matrix','teal','Reoriented grouping')]:
            c=mean_squared_hidden*f[f'p000_{layout}_axis1_noise_variance_coefficient']/50257/8192
            ax[1].bar(x+offset,c[channels],.36,color=color,label=label)
    ax[1].set_xticks(x,[str(i)for i in channels]);ax[1].set(yscale='log',xlabel='Hidden channel',ylabel='Contribution to mean logit variance',title='Analytic variance with BF16 hidden state fixed');ax[1].legend()
    fig.tight_layout();fig.savefig(dest/'causal_and_neurons.png',dpi=160);fig.savefig(dest/'causal_and_neurons.svg');plt.close(fig)
    print(json.dumps({'audit':report['status'],'logit_variance_reduction_factor':report['fixed_hidden_variance_reduction_factor'],'conditions':conditions},indent=2))

if __name__=='__main__':main()
