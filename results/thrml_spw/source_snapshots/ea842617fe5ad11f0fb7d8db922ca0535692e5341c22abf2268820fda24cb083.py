"""Explicitly provisional layer/neuron chart; never a five-seed quality gate."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 root=OUT/'lfm3_training_depth_v1';names=['selected_mean','selected_seed2026111801'];matrices=[];summaries=[]
 for name in names:
  summary=json.loads((root/name/'results.json').read_text());summaries.append(summary)
  with np.load(root/name/'channels.npz')as f:matrices.append({k:f[k].copy()for k in f.files})
 dest=OUT/'lfm3_first_seed_diagnostic_v1';dest.mkdir(exist_ok=True)
 baseline=json.loads((root/'BF16/results.json').read_text());s=matrices[1]['signal_SSE'];e=matrices[1]['error_SSE'];layers=[]
 for layer in range(24):
  top=np.argsort(e[layer])[-20:][::-1]
  layers.append({'layer':layer,'operator':baseline['layer_types'][layer],'relative_RMSE':float(np.sqrt(e[layer].sum()/s[layer].sum())),
    'top20_neurons':top.tolist(),'top20_fraction_of_SSE':float(e[layer,top].sum()/e[layer].sum())})
 fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
 for name,m in zip(['Controller mean','One sampled weight draw'],matrices):axes[0].plot(range(24),100*np.sqrt(m['error_SSE'].sum(1)/m['signal_SSE'].sum(1)),label=name,marker='.')
 axes[0].set(title='LFM3 training diagnostic: one provisional seed',ylabel='Block output relative RMS error (%)',xlabel='Block');axes[0].legend()
 heat=axes[1].imshow(np.log10(np.maximum(np.sqrt(e/s),1e-6)),aspect='auto',cmap='magma',interpolation='nearest');axes[1].set(xlabel='Hidden channel',ylabel='Block');fig.colorbar(heat,ax=axes[1],label='log10(relative RMS error)')
 fig.savefig(dest/'provisional_layers.png',dpi=180);fig.savefig(dest/'provisional_layers.svg');plt.close(fig)
 payload={'status':'provisional_one_seed_training_diagnostic_not_quality_gate','source_sha256':sha(__file__),'baseline_NLL':baseline['NLL'],'seed_NLL':summaries[1]['NLL'],
  'seed_PPL_ratio':float(np.exp(summaries[1]['NLL']-baseline['NLL'])),'seed':2026111801,'targets':8192,'layers':layers,
  'scope':'Propagated block error, not isolated causal sensitivity. One seed, training cohort, no heldout or general success claim.'}
 (dest/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items()if k!='layers'}))
if __name__=='__main__':main()
