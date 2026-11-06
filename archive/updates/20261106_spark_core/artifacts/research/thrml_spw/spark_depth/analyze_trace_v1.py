"""Aggregate paired training traces; descriptive diagnostics, not quality gates."""
import json,math,csv
from pathlib import Path
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'trace_v1';report=json.loads((root/'results.json').read_text());dest=OUT/'trace_analysis_v1';dest.mkdir(exist_ok=False)
    points={};channel_rows=[]
    for point in report['points']:
        name=point['name'];records=[r for r in report['records']if r['point']==name];modules={};heads=[];nll=[]
        for r in records:
            with np.load(r['statistics_file'],allow_pickle=False)as data:
                heads.append(data['attention_head_statistics']);nll.extend(data['token_NLL'])
                for module,info in r['modules'].items():
                    sums=data[module+'_channel_sums'];entry=modules.setdefault(module,{'sums':np.zeros_like(sums),'rows':0})
                    entry['sums'][:6]+=sums[:6];entry['sums'][6:8]=np.maximum(entry['sums'][6:8],sums[6:8]);entry['sums'][8]+=sums[8];entry['rows']+=info['rows']
        module_report={}
        for module,entry in modules.items():
            s=entry['sums'];n=entry['rows'];error=s[5]/n;signal=s[3]/n;relative=np.sqrt(error/np.maximum(signal,1e-30));bias=(s[4]/n)**2
            order=np.argsort(error)[::-1];total=error.sum()
            module_report[module]={'RMSE':float(np.sqrt(error.mean())),'reference_RMS':float(np.sqrt(signal.mean())),
                'relative_RMSE':float(np.sqrt(error.sum()/max(signal.sum(),1e-30))),
                'squared_mean_error_fraction':float(bias.sum()/max(total,1e-30)),
                'largest_1pct_channels_error_fraction':float(error[order[:max(1,math.ceil(len(order)*.01))]].sum()/max(total,1e-30)),
                'top10_error_channels':[{'channel':int(i),'RMSE':float(np.sqrt(error[i])),'relative_RMSE':float(relative[i]),'mean_error':float(s[4,i]/n),'max_abs_error':float(s[7,i])}for i in order[:10]]}
            if module!='lm_head':
                for i in range(len(error)):channel_rows.append([name,module,i,float(np.sqrt(error[i])),float(np.sqrt(signal[i])),float(relative[i]),float(s[4,i]/n)])
        metric={k:float(np.average([r['metrics'][k]for r in records],weights=[r['metrics']['targets']for r in records]))for k in records[0]['metrics']if k!='targets'}
        metric['targets']=sum(r['metrics']['targets']for r in records);metric['perplexity']=math.exp(metric['mean_NLL'])
        points[name]={'metrics':metric,'modules':module_report,'attention_head_stats_mean_across_windows':np.mean(heads,axis=0).tolist()}
    summary={'scope':'Eight GPT2 training articles,129tokens each. Descriptive paired diagnostics. Gaussian variance proxy is not native pbit or heldout evidence.',
        'source_sha256':sha(__file__),'trace_report_sha256':sha(root/'results.json'),'points':points}
    (dest/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (dest/'channels.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['point','module','channel','RMSE','reference_RMS','relative_RMSE','mean_error']);writer.writerows(channel_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('GPT-2 on DGX Spark: training-only diagnostic, 1,024 targets')
    labels={'finite_mean_G128':'Finite-control mean','Gaussian_proxy_G128_T8192_seed2026110611':'Gaussian variance proxy (T=8,192)'}
    for point,label in labels.items():
        modules=points[point]['modules'];axes[0,0].plot(range(12),[modules[f'transformer.h.{i}']['relative_RMSE']for i in range(12)],marker='o',label=label)
    axes[0,0].set(xlabel='Transformer block',ylabel='Residual-output relative RMSE',title='Observed error propagation (not causal sensitivity)');axes[0,0].legend(fontsize=8)
    gaussian=points['Gaussian_proxy_G128_T8192_seed2026110611'];head=np.asarray(gaussian['attention_head_stats_mean_across_windows'])[:,1,:]
    im=axes[0,1].imshow(head,aspect='auto');fig.colorbar(im,ax=axes[0,1],label='Mean KL(reference || proxy)');axes[0,1].set(xlabel='Attention head',ylabel='Transformer block',title='Attention distribution changes')
    for suffix,label in [('ln_1','Pre-attention norm'),('attn.c_attn','QKV'),('mlp.act','MLP activation'),('mlp.c_proj','MLP output')]:
        axes[1,0].plot(range(12),[gaussian['modules'][f'transformer.h.{i}.{suffix}']['relative_RMSE']for i in range(12)],marker='.',label=label)
    axes[1,0].set(xlabel='Transformer block',ylabel='Relative RMSE',title='Module outputs under Gaussian proxy');axes[1,0].legend(fontsize=8)
    survey=json.loads((OUT/'weight_survey_v1/results.json').read_text());totals={}
    for record in survey['parameters']:
        for condition in record['conditions']:
            key=(condition['layout'],condition['group_size']);totals[key]=totals.get(key,0)+record['elements']*condition['predicted_noise_RMSE_T8192']**2
        if len(record['shape'])==1:
            for condition in record['conditions']:
                key=('transposed_matrix',condition['group_size']);totals[key]=totals.get(key,0)+record['elements']*condition['predicted_noise_RMSE_T8192']**2
    for layout,label in [('storage_flat','Original storage'),('transposed_matrix','Transposed matrices')]:axes[1,1].plot([16,32,64,128],[totals[layout,g]for g in [16,32,64,128]],marker='o',label=label)
    axes[1,1].set(xlabel='Weights per range group',ylabel='Predicted total weight squared error',title='Weight-variance prediction only; no accuracy claim');axes[1,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'diagnostics.png',dpi=160);fig.savefig(dest/'diagnostics.svg');plt.close(fig)
    print(json.dumps({k:v['metrics']for k,v in points.items()},indent=2))

if __name__=='__main__':main()
