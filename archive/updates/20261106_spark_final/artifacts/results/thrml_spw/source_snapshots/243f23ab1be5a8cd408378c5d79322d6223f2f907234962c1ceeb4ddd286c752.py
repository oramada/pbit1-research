"""Summarize only the two completed, audited and replayed model cohorts."""
from pathlib import Path
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    files={'GPT2_audit':OUT/'benchmark_audit_v1/results.json','GPT2_replay':OUT/'benchmark_replay_v1/all_results.json',
        'DistilBERT_audit':OUT/'distilbert_benchmark_audit_v1/results.json','DistilBERT_replay':OUT/'distilbert_replay_v1/results.json'}
    data={k:json.loads(v.read_text())for k,v in files.items()};assert len(data['GPT2_replay']['cases'])==len(data['DistilBERT_replay']['cases'])==12
    g=data['GPT2_audit'];d=data['DistilBERT_audit'];models=[]
    for label,results,metric in [('GPT2',g,'PPL'),('DistilBERT',d,'CE')]:
        if label=='GPT2':
            stats=results['statistics']['selected'];baseline=results['cases'][0]['NLL'];values=np.array([r['NLL']for r in results['cases']if r['case']['kind']=='selected']);quality=stats['primary_2pct_pass']
        else:
            stats=results['statistics']['selected'];baseline=results['cases'][0]['means']['CE'];values=np.array([r['means']['CE']for r in results['cases']if r['case']['kind']=='selected']);quality=stats['joint_pass']
        delta=values-baseline;half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
        entry={'model':label,'seed_count':5,'perplexity_ratio':float(np.exp(delta.mean())),'MC95_perplexity_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'passes_own_frozen_quality_gates':bool(quality)}
        if label=='DistilBERT':entry.update(BF16_accuracy=results['cases'][0]['means']['correct'],mean_selected_accuracy=float(np.mean([r['means']['correct']for r in results['cases']if r['case']['kind']=='selected'])),accuracy_decline_upper95=stats['accuracy_decline_upper_MC95'])
        models.append(entry)
    dest=OUT/'cross_model_summary_v1';dest.mkdir(exist_ok=False)
    result={'source_sha256':sha(__file__),'evidence_sha256':{k:sha(v)for k,v in files.items()},'models':models,'both_tested_models_pass':all(x['passes_own_frozen_quality_gates']for x in models),
        'scope':'Two testedarchitectures/tasks, ideal independent Bernoulli R1reconstruction, onepbitperlearnedscalar,12bitcontrols,T32768. Previouslyinspectedbenchmarkre-evaluations. Notuniversalnumericalequality,physicalhardwarebenefit,orfullmodelnativeTHRMLvalidation.'}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8))
    for i,m in enumerate(models):
        mean=m['perplexity_ratio'];lo,hi=m['MC95_perplexity_ratio_interval'];ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=8,color='teal')
    ax.axhline(1,color='gray',linestyle=':',label='BF16');ax.axhline(1.02,color='orange',linestyle='--',label='Frozen 2% probability-quality threshold')
    ax.set_xticks([0,1],['GPT-2 / WikiText2','DistilBERT / IMDb']);ax.set(ylabel='Categorical perplexity / BF16',title='Completed, audited five-seed model cohorts\nK1, 12-bit controls, 32,768 samples');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'tested_model_quality.png',dpi=160);fig.savefig(dest/'tested_model_quality.svg')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
