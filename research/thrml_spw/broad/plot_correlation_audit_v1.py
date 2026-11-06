"""Export a bounded correlation-cost figure from independently audited tables."""
from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/thrml_spw');audit=json.loads((root/'broad_verification_20261105/correlation_independent_audit_v1/results.json').read_text())
    table=root/'correlated_dot/dot_product_mse.csv'
    assert hashlib.sha256(table.read_bytes()).hexdigest()==audit['input_sha256']['dot_product_mse.csv']
    with table.open()as f:rows=list(csv.DictReader(f))
    labels={'positive_uniform':'Positive uniform','mixed_alternating':'Alternating signs','mixed_graded':'Graded mixed signs'}
    fig,ax=plt.subplots(figsize=(7.7,4.5),layout='constrained')
    for name,label in labels.items():
        subset=[r for r in rows if int(r['n_spins'])==4 and float(r['coupling_J'])==.8 and r['input_vector']==name]
        assert len(subset)==4;subset.sort(key=lambda r:int(r['samples_T']))
        ax.plot([int(r['samples_T']) for r in subset],[float(r['exact_to_naive_ratio']) for r in subset],marker='o',label=label,linewidth=2)
    ax.axhline(1,color='#667085',linestyle='--',linewidth=1,label='Independent estimate')
    ax.set_xscale('log',base=2);ax.set_xticks([1,16,64,256],['1','16','64','256'])
    ax.set_xlabel('Samples averaged, T');ax.set_ylabel('Exact variance / independent variance')
    ax.set_title('Correlation changes sampling cost',loc='left',weight='bold',pad=12)
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    fig.text(.02,-.04,'Four-spin THRML Ising chain, J = 0.8; marginals calibrated. Exact stationary calculation.\nSmall-operator result; no physical hardware or large-model performance claim.',fontsize=9,color='#475467')
    out=root/'broad_verification_20261105/correlation_independent_audit_v1/correlation_limits.svg'
    fig.savefig(out,bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    print(out)

if __name__=='__main__':main()
