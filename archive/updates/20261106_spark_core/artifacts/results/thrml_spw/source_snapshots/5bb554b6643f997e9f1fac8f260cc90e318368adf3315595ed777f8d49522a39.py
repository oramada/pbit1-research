"""CPU-only baseline activation tails/correlations at all 24 LFM3 block outputs."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 root=OUT/'lfm3_training_depth_v1';result=json.loads((root/'results.json').read_text());rng=np.random.default_rng(2026112501)
 error=[]
 for case in result['cases'][2:]:
  with np.load(root/case['case']['name']/'channels.npz')as f:error.append(f['error_SSE'].copy())
 e=np.mean(error,axis=0);selected=np.argsort(e,axis=1)[:,-64:];random=np.array([rng.choice(2048,64,replace=False)for _ in range(24)])
 sums=np.zeros((24,2048));second=np.zeros_like(sums);fourth=np.zeros_like(sums);maximum=np.zeros_like(sums);cross=np.zeros((2,24,64,64));count=0
 for row in range(32):
  with np.load(root/'BF16'/f'layer_states{row:02d}.npz')as f:
   for layer in range(24):
    # BF16 is the high 16 bits of its exact FP32 value.
    a=(f[f'layer{layer:02d}'].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(-1,2048)
    sums[layer]+=a.sum(0);second[layer]+=(a*a).sum(0);fourth[layer]+=(a**4).sum(0);maximum[layer]=np.maximum(maximum[layer],np.abs(a).max(0))
    for group,indices in enumerate([selected[layer],random[layer]]):
     subset=a[:,indices];cross[group,layer]+=subset.T@subset
   count+=a.shape[0]
 with np.load(root/'BF16'/'channels.npz')as f:np.testing.assert_allclose(second,f['signal_SSE'],rtol=1e-13)
 means=sums/count;mom2=second/count;rows=[];correlations=np.zeros_like(cross)
 for layer in range(24):
  record={'layer':layer,'operator':result['cases'][0]['layer_types'][layer],
    'max_to_median_neuron_RMS':float(np.sqrt(mom2[layer].max()/np.median(mom2[layer]))),
    'energy_participation_ratio':float(second[layer].sum()**2/np.sum(second[layer]**2)),
    'max_uncentered_fourth_moment_ratio':float(np.max(fourth[layer]*count/(second[layer]**2))),
    'top64_error_fraction':float(e[layer,selected[layer]].sum()/e[layer].sum())}
  for group,(label,indices)in enumerate([('top_error',selected[layer]),('random',random[layer])]):
   cov=cross[group,layer]/count-np.outer(means[layer,indices],means[layer,indices]);std=np.sqrt(np.maximum(np.diag(cov),0));corr=cov/np.maximum(np.outer(std,std),1e-30);correlations[group,layer]=corr
   upper=corr[np.triu_indices(64,1)];record[label+'_mean_abs_activation_correlation']=float(np.mean(np.abs(upper)))
  rows.append(record)
 dest=OUT/'lfm3_neuron_structure_v1';dest.mkdir(exist_ok=False)
 with(dest/'moments.npz').open('xb')as f:np.savez_compressed(f,means=means,second_moments=mom2,fourth_moments=fourth/count,max_abs=maximum,top64_error_channels=selected,random64_channels=random,activation_correlations=correlations)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');axes[0].plot(range(24),[x['energy_participation_ratio']for x in rows],marker='.');axes[0].set(xlabel='Block',ylabel='Effective channels by signal energy',title='Baseline activation concentration')
 for label in ['top_error','random']:axes[1].plot(range(24),[x[label+'_mean_abs_activation_correlation']for x in rows],marker='.',label=label.replace('_',' '))
 axes[1].set(xlabel='Block',ylabel='Mean absolute off-diagonal correlation',title='Activation correlation in 64-channel subsets');axes[1].legend()
 fig.savefig(dest/'neuron_structure.png',dpi=180);fig.savefig(dest/'neuron_structure.svg');plt.close(fig)
 output={'status':'all24_block_baseline_activation_structures_measured','source_sha256':sha(__file__),'training_result_sha256':sha(root/'results.json'),'positions_per_layer':count,'rows':rows,
  'scope':'Original BF16 training activations. These are activation correlations, not stochastic-error correlations. Top-error channel selection uses the completed five-seed training diagnostic. Not causal attribution or heldout evidence.'}
 (dest/'results.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'status':output['status'],'positions_per_layer':count,'participation_ratio_range':[min(x['energy_participation_ratio']for x in rows),max(x['energy_participation_ratio']for x in rows)]}))
if __name__=='__main__':main()
