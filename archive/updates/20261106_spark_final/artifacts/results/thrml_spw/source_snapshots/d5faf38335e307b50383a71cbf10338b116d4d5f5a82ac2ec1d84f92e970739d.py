"""Plot audited cumulative block substitutions without claiming monotonic damage."""
import json
import numpy as np
from .prepare_v1 import OUT,sha


def main():
    file=OUT/'gpt2_cumulative_v1_audit.json';audit=json.loads(file.read_text());assert audit['producer_result_sha256']==sha(OUT/'gpt2_cumulative_v1/results.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for side in ['prefix','suffix']:
        rows=[audit['curves'][side+str(n)]for n in [3,6,9]]+[audit['curves']['all_blocks']]
        mean=np.array([0]+[x['delta_NLL']for x in rows]);low=np.array([0]+[x['MC95_delta_NLL'][0]for x in rows]);high=np.array([0]+[x['MC95_delta_NLL'][1]for x in rows])
        ax.errorbar([0,3,6,9,12],mean,yerr=[mean-low,high-mean],marker='o',capsize=3,label=side+' blocks')
    ax.axhline(0,color='gray',linestyle=':');ax.set(xlabel='Stochastic transformer blocks (nonblock parameters deterministic)',ylabel='Mean NLL change from CPU reference',title='Cumulative GPT-2 substitutions: five draws, 1024 training targets');ax.legend()
    dest=OUT/'gpt2_cumulative_summary_v1';dest.mkdir(exist_ok=True);fig.savefig(dest/'cumulative_blocks.png',dpi=180);fig.savefig(dest/'cumulative_blocks.svg');plt.close(fig)
    (dest/'results.json').write_text(json.dumps({'source_sha256':sha(__file__),'audit_sha256':sha(file),'curves':audit['curves'],'scope':audit['scope']},indent=2)+'\n')
if __name__=='__main__':main()
