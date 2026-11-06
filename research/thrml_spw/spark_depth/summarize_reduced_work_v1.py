"""Compare accepted uniform and lower-work GPT2 policies at the same quality margin."""
import json
import numpy as np
from .prepare_v1 import OUT,sha


def main():
    a=json.loads((OUT/'reduced_work_gpu_v1_audit.json').read_text());replay=json.loads((OUT/'reduced_work_replay_v1/results.json').read_text());assert a['result_sha256']==replay['benchmark_result_sha256']==sha(OUT/'reduced_work_gpu_v1/results.json')
    assert len(replay['cases'])==6 and sum(len(x['replays'])for x in replay['cases'])==18
    assert all(y['all_outputs_bitwise_equal']for x in replay['cases']for y in x['replays'])
    old=json.loads((OUT/'benchmark_v1/results.json').read_text());uniform=old['statistics']['selected'];lower=a['statistics'];assert uniform['pass']and lower['pass']
    x=np.array([1.,a['observation_fraction']]);y=np.array([uniform['PPL_ratio'],lower['PPL_ratio']]);bounds=np.array([uniform['MC95_t_PPL_ratio_interval'],lower['MC95_t4_PPL_ratio_interval']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].bar(['Uniform','Sensitivity-based'],x,color=['#4477aa','#228833']);axes[0].set(ylabel='Nominal observations / uniform budget',title='41.7% fewer sampling observations');axes[0].set_ylim(0,1.15)
    axes[1].errorbar(x,y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt='o',capsize=4,color='#228833');axes[1].axhline(1.02,color='#aa4444',ls='--',label='Fixed 2% margin');axes[1].axhline(1,color='gray',ls=':');axes[1].set(xlabel='Nominal observations / uniform budget',ylabel='Perplexity ratio to BF16',title='Both policies meet the quality criterion');axes[1].legend(loc='upper right')
    dest=OUT/'reduced_work_summary_v1';dest.mkdir(exist_ok=True);fig.savefig(dest/'sampling_quality_tradeoff.png',dpi=180);fig.savefig(dest/'sampling_quality_tradeoff.svg');plt.close(fig)
    result={'source_sha256':sha(__file__),'audit_sha256':sha(OUT/'reduced_work_gpu_v1_audit.json'),'replay_sha256':sha(OUT/'reduced_work_replay_v1/results.json'),'uniform':uniform,'reduced':lower,'observation_fraction':a['observation_fraction'],'maximum_samples':a['maximum_samples'],'extra_range_metadata_bytes':a['extra_range_metadata_bytes'],
     'scope':'Separate five-seed cohorts on the same previously inspected full task. Both satisfy the same predefined margin; not proof of identical accuracy, physical energy savings, or lower inference latency. R1 reconstruction budgets, one logical pbit per all124439808scalars.'}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
