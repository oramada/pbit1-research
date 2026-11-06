"""Operator-aware noise predictions from clean activation second moments."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    dest=OUT/'activation_variance_v1';dest.mkdir(exist_ok=False)
    survey=json.loads((OUT/'weight_survey_v1/results.json').read_text());parameters={p['name']:p for p in survey['parameters']}
    operators=[('lm_head','transformer.ln_f','transformer.wte.weight',1)]
    for i in range(12):
        b=f'transformer.h.{i}';operators.extend([(b+'.attn.c_attn',b+'.ln_1',b+'.attn.c_attn.weight',0),(b+'.mlp.c_fc',b+'.ln_2',b+'.mlp.c_fc.weight',0),(b+'.mlp.c_proj',b+'.mlp.act',b+'.mlp.c_proj.weight',0)])
    input_squares={output:None for output,_,_,_ in operators};output_squares={output:0. for output,_,_,_ in operators};tokens=0
    for row in range(8):
        with np.load(OUT/'trace_v1/original_BF16'/f'row{row}.npz',allow_pickle=False)as f:
            for output,input_module,weight,axis in operators:
                a=f[input_module][0,:-1].astype(np.float64);b=f[output][0,:-1].astype(np.float64);sq=(a*a).sum(0)
                input_squares[output]=sq if input_squares[output]is None else input_squares[output]+sq
                output_squares[output]+=float((b*b).sum())
            tokens+=len(a)
    reports=[]
    with np.load(OUT/'weight_survey_v1/axis_statistics.npz',allow_pickle=False)as f:
        for output,source,weight,axis in operators:
            param=parameters[weight];moments=input_squares[output]/tokens;conditions={}
            for layout in ['storage_flat','transposed_matrix']:
                v=f"p{param['index']:03d}_{layout}_axis{axis}_noise_variance_coefficient";per_feature=f[v]/8192;assert per_feature.shape==moments.shape
                variance=float(moments@per_feature);reference=output_squares[output]/tokens
                conditions[layout]={'predicted_sum_output_variance_per_token':variance,'reference_sum_squared_output_per_token':reference,
                    'predicted_relative_output_RMSE':float(np.sqrt(variance/reference)),
                    'top_noise_input_features':np.argsort(moments*per_feature)[-8:][::-1].tolist()}
            ratio=conditions['transposed_matrix']['predicted_sum_output_variance_per_token']/conditions['storage_flat']['predicted_sum_output_variance_per_token']
            reports.append({'operator':output,'weight':weight,'weight_index':param['index'],'input_moment_source':source,
                'preferred_by_local_variance':'transposed_matrix'if ratio<1 else'storage_flat','transposed_over_storage_predicted_variance':ratio,'conditions':conditions})
    result={'source_sha256':sha(__file__),'scope':'Conditional independent weight-noise prediction with cleanBF16activationsecondmoments. T8192,G128. Ignores upstreamchanges,nonlinearpropagation,BF16roundingandoutputcovariance. Local variance preference is a candidate selector, not proof of task quality. Attentionoutputprojection omitted because itsinput wasnotrecorded.',
        'tokens':tokens,'operators':reports}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for suffix,label in [('attn.c_attn','QKV projection'),('mlp.c_fc','MLP expansion'),('mlp.c_proj','MLP contraction')]:
        ratio=[next(r for r in reports if r['operator']==f'transformer.h.{i}.{suffix}')['transposed_over_storage_predicted_variance']for i in range(12)]
        ax.plot(range(12),ratio,marker='o',label=label)
    ax.axhline(1,color='gray',linestyle='--');ax.set(xlabel='Transformer block',ylabel='Predicted output variance: reoriented / original',title='Grouping direction matters differently at each operator',yscale='log');ax.legend();fig.tight_layout();fig.savefig(dest/'operator_preferences.png',dpi=160);fig.savefig(dest/'operator_preferences.svg')
    print(json.dumps({'operators':len(reports),'prefer_transposed':sum(r['preferred_by_local_variance']=='transposed_matrix'for r in reports),'head_variance_ratio':reports[0]['transposed_over_storage_predicted_variance']}))

if __name__=='__main__':main()
