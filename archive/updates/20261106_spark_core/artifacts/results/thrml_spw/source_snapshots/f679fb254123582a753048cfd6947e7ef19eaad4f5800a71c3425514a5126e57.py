"""Post-evaluation output robustness diagnostics; never a policy-selection set."""
from pathlib import Path
import json
import numpy as np
from scipy.special import softmax,logsumexp
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'distilbert_benchmark_v1';audit_file=OUT/'distilbert_benchmark_audit_v1/results.json';audit=json.loads(audit_file.read_text());assert audit['status'].startswith('all12cases_')
    plan=json.loads((root/'protocol.json').read_text())['plan'];dest=OUT/'distilbert_output_diagnostics_v1';dest.mkdir(exist_ok=False)
    def read(name):
        logits=[];labels=[]
        for index in range((25000+plan['batch_size']-1)//plan['batch_size']):
            with np.load(root/name/f'batch{index:04d}.npz')as f:logits.append(f['logits'].copy());labels.append(f['labels'].copy())
        return np.concatenate(logits),np.concatenate(labels)
    z,y=read('BF16');prob=softmax(z.astype(np.float64),axis=1);baseline=z.argmax(1);margin=np.abs(prob[:,1]-prob[:,0]);correct=baseline==y
    records=[];edges=np.linspace(0,1,11)
    for kind in ['selected','matched_flat']:
        points=[]
        for c in plan['cases']:
            if c['kind']!=kind:continue
            logits,labels=read(c['name']);assert np.array_equal(labels,y);points.append(logits)
        points=np.stack(points).astype(np.float64);pred=points.argmax(-1);flip=pred!=baseline
        logp=points-logsumexp(points,axis=-1,keepdims=True);base_log=prob.clip(1e-300).astype(float);base_log=np.log(base_log)
        ce=-logp[:,np.arange(25000),y];base_ce=-base_log[np.arange(25000),y]
        bins=[]
        for i in range(10):
            select=(margin>=edges[i])&(margin<=edges[i+1]if i==9 else margin<edges[i+1]);count=int(select.sum())
            bins.append({'low_margin':float(edges[i]),'high_margin':float(edges[i+1]),'reviews':count,'mean_flip_fraction_across_seeds':float(flip[:,select].mean())if count else None})
        records.append({'method':kind,'seed_count':len(points),'mean_flip_fraction':float(flip.mean()),'margin_bins':bins,
            'baseline_correct_became_wrong_fraction':float(((pred!=y)&correct).mean()),'baseline_wrong_became_correct_fraction':float(((pred==y)&~correct).mean()),
            'mean_CE_change_baseline_correct':float((ce-base_ce)[:,correct].mean()),'mean_CE_change_baseline_wrong':float((ce-base_ce)[:,~correct].mean())})
    report={'source_sha256':sha(__file__),'audit_sha256':sha(audit_file),'records':records,
        'scope':'Posthocdescriptiveoutputdiagnostics oncompletedtestset; do notusefornewthresholdselection. Reviewbinsarecorrelatedwithinmodeldraw, so no IIDreview-basedconfidenceintervalsareclaimed.'}
    (dest/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for r in records:
        ax.plot([(.5*(b['low_margin']+b['high_margin']))for b in r['margin_bins']],[b['mean_flip_fraction_across_seeds']for b in r['margin_bins']],marker='o',label=r['method'])
    ax.set(xlabel='Original BF16 binary probability margin',ylabel='Prediction flip fraction across five seeds',title='DistilBERT: prediction changes versus original margin');ax.legend();fig.tight_layout();fig.savefig(dest/'margin_sensitivity.png',dpi=160);fig.savefig(dest/'margin_sensitivity.svg')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
