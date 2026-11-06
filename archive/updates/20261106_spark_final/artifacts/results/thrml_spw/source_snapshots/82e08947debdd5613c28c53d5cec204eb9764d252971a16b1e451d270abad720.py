"""Plot unvalidated, gradient-based sample-allocation proposals."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def category(name):
    if '.h.'in name:return 'block '+name.split('.h.')[1].split('.')[0]
    if '.wte.'in name:return 'embedding/head'
    if '.wpe.'in name:return 'position'
    return 'final norm'

def main():
    source=OUT/'loss_gradient_variance_v1/results.json';r=json.loads(source.read_text());dest=OUT/'gradient_allocation_summary_v1';dest.mkdir(exist_ok=False)
    labels=['embedding/head','position']+['block '+str(i)for i in range(12)]+['final norm'];summary={}
    for p in r['allocation_proposals']:
        summary[p['name']]={}
        for label in labels:
            rows=[x for x in p['parameters']if category(x['name'])==label];n=sum(x['parameters']for x in rows);obs=sum(x['parameters']*x['samples']for x in rows)
            summary[p['name']][label]={'parameters':n,'weighted_mean_samples':obs/n,'observations':obs,'predicted_loss_fluctuation_variance':sum(x['proxy_coefficient']/x['samples']for x in rows)}
    (dest/'results.json').write_text(json.dumps({'source_sha256':sha(__file__),'proposal_results_sha256':sha(source),'summary':summary,
        'scope':'Predicted allocation only; constant physicalpbitcount. Trainedon8shortarticles usingfirst-orderlossgradient proxy; notvalidatedneuralqualityorphysicalworkmeasurement.'},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5))
    for name,label in [('existing_directions','Keep existing directions'),('gradient_proxy_directions','Also choose directions by gradient proxy')]:
        ax.plot(range(len(labels)),[summary[name][x]['weighted_mean_samples']for x in labels],marker='o',label=label)
    ax.axhline(32768,color='gray',linestyle='--',label='Uniform reference');ax.set_xticks(range(len(labels)),labels,rotation=50,ha='right');ax.set(yscale='log',ylabel='Parameter-weighted mean samples',title='Training-only sample allocation proposal — not a quality result');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'allocation_proposal.png',dpi=160);fig.savefig(dest/'allocation_proposal.svg')
    print(json.dumps({p['name']:p['versus_uniform_32768_cost']for p in r['allocation_proposals']}))

if __name__=='__main__':main()
