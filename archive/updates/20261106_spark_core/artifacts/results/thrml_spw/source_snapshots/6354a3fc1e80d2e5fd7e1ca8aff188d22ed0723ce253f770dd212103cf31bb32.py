"""Validate and plot the training-only DistilBERT operator diagnostics."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'distilbert_cpu_diagnostics_v1';report=json.loads((root/'results.json').read_text());dest=OUT/'distilbert_diagnostic_summary_v1';dest.mkdir(exist_ok=False)
    assert sha(root/'channel_statistics.npz')==report['channel_statistics_sha256']
    with np.load(OUT/'distilbert_preparation_v1/calibration.npz',allow_pickle=False)as f:tokens=int(f['attention_mask'].sum())
    checks=[]
    with np.load(root/'channel_statistics.npz',allow_pickle=False)as a:
        for m in report['modules']:
            name=m['module'];count=m['observations'];squares=a[name+'.output_sumsq'];assert count in (64,tokens)
            assert np.isfinite(squares).all()and np.all(squares>=0)
            assert np.isclose(np.sqrt(squares.sum()/count/len(squares)),m['output_RMS'])
            if m['type']=='Embedding':assert a[name+'.lookup_counts'].sum()==count
            if 'variants'in m:
                variants=m['variants'];assert m['preference']==min(variants,key=lambda key:variants[key]['predicted_sum_output_variance_T8192'])
                for d in variants.values():
                    assert np.isclose(d['predicted_relative_output_RMSE_T8192']**2,d['predicted_sum_output_variance_T8192']/(squares.sum()/count))
            checks.append(name)
    families=['q_lin','k_lin','v_lin','out_lin','lin1','lin2'];matrix=[]
    for layer in range(6):
        row=[]
        for family in families:
            name=f'distilbert.transformer.layer.{layer}.'+('ffn.'if family.startswith('lin')else'attention.')+family
            module=next(x for x in report['modules']if x['module']==name);v=module['variants'];row.append(v['transposed_matrix']['predicted_sum_output_variance_T8192']/v['storage_flat']['predicted_sum_output_variance_T8192'])
        matrix.append(row)
    summary={'status':'validated_all53_operator_channel_summaries_and40_direction_choices','source_sha256':sha(__file__),'results_sha256':sha(root/'results.json'),
        'tokens':tokens,'review_count':64,'unique_parameters':report['unique_parameter_count'],'preferred_transposed':sum(v=='transposed_matrix'for v in report['preferences'].values()),
        'preferred_original':sum(v=='storage_flat'for v in report['preferences'].values()),'family_order':families,'transposed_over_original_predicted_variance':matrix,
        'scope':'Clean FP32 CPU activations with BF16-roundedweights ontrainingreviews. Operatorvariancepredictions, nottaskaccuracy orstochasticinference.'}
    (dest/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5));m=np.array(matrix);im=ax.imshow(np.log2(m),cmap='RdBu_r',vmin=-1.7,vmax=1.7,aspect='auto');ax.set_xticks(range(6),families);ax.set_yticks(range(6));ax.set(xlabel='Operator',ylabel='DistilBERT layer',title='Grouping direction depends on the operator\nCell values: predicted variance, transposed / original')
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(im,ax=ax,label='log2 variance ratio; blue favors transposed');fig.tight_layout();fig.savefig(dest/'operator_direction.png',dpi=160);fig.savefig(dest/'operator_direction.svg')
    print(json.dumps({k:v for k,v in summary.items()if k!='transposed_over_original_predicted_variance'},indent=2))

if __name__=='__main__':main()
