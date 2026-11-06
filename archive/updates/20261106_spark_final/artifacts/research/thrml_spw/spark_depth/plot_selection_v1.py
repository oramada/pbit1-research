"""Display calibration selection separately from unfinished benchmark results."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads((OUT/'profile_calibration_audit_v1.json').read_text());b=json.loads((OUT/'profile_extension_combined_v1.json').read_text());dest=OUT/'selection_figures_v1';dest.mkdir(exist_ok=False)
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    labels={'embedding_only_T8192':'Embedding-only direction','operator_T8192':'Operator directions','operator_body_T32768':'More samples outside embedding','operator_all_T32768':'More samples everywhere'}
    for i,(name,p)in enumerate(a['calibration'].items()):
        lo,hi=p['MC95_t_PPL_ratio_interval'];m=p['PPL_ratio'];ax[0].errorbar(p['observation_cost_ratio'],m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=5,label=labels[name])
    for i,(name,p)in enumerate(b['summaries'].items()):
        lo,hi=p['MC95_t9_PPL_ratio_interval'];m=p['PPL_ratio'];ax[1].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color='teal'if p['qualified']else'tomato')
    ax[1].set_xticks([0,1],['Outside embedding','Everywhere'])
    for axis in ax:
        axis.axhline(1,color='gray',linestyle=':');axis.axhline(1.02,color='orange',linestyle='--');axis.set_ylabel('Calibration perplexity / BF16')
    ax[0].set(xlabel='Bernoulli observation cost / original budget',title='Initial five seeds: all fail upper-bound rule');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Where the larger sample budget is assigned',title='Ten seeds: one policy qualifies for benchmark')
    fig.suptitle('GPT-2 calibration only — full benchmark still running');fig.tight_layout();fig.savefig(dest/'calibration_selection.png',dpi=160);fig.savefig(dest/'calibration_selection.svg')
    (dest/'manifest.json').write_text(json.dumps({'source_sha256':sha(__file__),'five_seed_audit_sha256':sha(OUT/'profile_calibration_audit_v1.json'),'ten_seed_results_sha256':sha(OUT/'profile_extension_combined_v1.json')},indent=2)+'\n')

if __name__=='__main__':main()
