"""Plot audited input-averaged conditional sampling-error covariance."""
import json
import numpy as np
from .prepare_v1 import OUT,sha


def main():
    root=OUT/'lfm3_error_covariance_v2';r=json.loads((root/'results.json').read_text());audit=json.loads((OUT/'lfm3_error_covariance_v2_audit.json').read_text());assert audit['result_sha256']==sha(root/'results.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');summary=[]
    for group,label in [(0,'Top-error 64 channels'),(1,'Fixed random 64 channels')]:
        rows=sorted([x for x in r['summaries']if x['group']==group],key=lambda x:x['layer']);assert len(rows)==24
        corr=[x['mean_absolute_error_correlation']for x in rows];rank=[x['covariance_participation_ratio']for x in rows]
        axes[0].plot(range(24),corr,marker='.',label=label);axes[1].plot(range(24),rank,marker='.',label=label)
        summary.append({'group':group,'mean_abs_correlation_range':[min(corr),max(corr)],'participation_ratio_range':[min(rank),max(rank)]})
    axes[0].set(xlabel='LFM3 block',ylabel='Mean absolute off-diagonal correlation',title='Conditional sampling-error correlation')
    axes[1].set(xlabel='LFM3 block',ylabel='Covariance participation ratio (64 coordinates)',title='Sampling-error covariance concentration');axes[0].legend();axes[1].legend()
    dest=OUT/'lfm3_error_covariance_summary_v1';dest.mkdir(exist_ok=True);fig.savefig(dest/'conditional_error_covariance.png',dpi=180);fig.savefig(dest/'conditional_error_covariance.svg');plt.close(fig)
    result={'source_sha256':sha(__file__),'audit_sha256':sha(OUT/'lfm3_error_covariance_v2_audit.json'),'groups':summary,'scope':r['scope']+' Correlations are derived from covariance averaged across input positions; these are descriptive subset statistics, not population confidence bounds.'}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
