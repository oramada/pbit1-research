"""Independent scoring and paired intervention plots for LFM3 causal study v2."""
import csv,json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 root=OUT/'lfm3_causal_depth_v2';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
 replay=json.loads((root/'replay_results.json').read_text());assert len(replay['replays'])==6 and all(x['all256_NLL_values_bitwise_match']for x in replay['replays'])
 rows=[]
 for case in p['cases']:
  folder=root/case['name'];saved=json.loads((folder/'results.json').read_text());losses=[]
  for row in range(8):
   with np.load(folder/f'row{row:02d}.npz')as f:
    assert f['token_NLL'].shape==(128,)and f['input_ids'].shape==(129,)and np.isfinite(f['token_NLL']).all();losses.extend(f['token_NLL'])
  nll=float(np.mean(losses));assert nll==saved['NLL']and saved['targets']==1024
  with np.load(folder/'channels.npz')as f:
   error=f['error_SSE'];signal=f['signal_SSE'];assert error.shape==signal.shape==(24,2048)and(error>=0).all()and(signal>0).all()
   np.testing.assert_allclose(np.sqrt(error.sum(1)/signal.sum(1)),saved['layer_relative_RMSE'],rtol=1e-14)
  rows.append({'name':case['name'],'NLL':nll,'stochastic_parameters':saved['stochastic_parameters']})
 lookup={x['name']:x for x in rows};assert lookup['BF16']['stochastic_parameters']==0 and lookup['all_K1']['stochastic_parameters']==1728801536
 layers=[]
 for layer in range(24):
  isolate=lookup[f'isolated_layer{layer:02d}'];rescue=lookup[f'rescued_layer{layer:02d}'];assert isolate['stochastic_parameters']+rescue['stochastic_parameters']==1728801536
  layers.append({'layer':layer,'isolated_delta_NLL':isolate['NLL']-lookup['BF16']['NLL'],'rescue_NLL_improvement':lookup['all_K1']['NLL']-rescue['NLL']})
 dest=OUT/'lfm3_causal_depth_analysis_v1';dest.mkdir(exist_ok=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,4),layout='constrained');x=np.arange(24)
 ax.bar(x-.2,[x['isolated_delta_NLL']for x in layers],width=.4,label='Only this block stochastic: NLL change')
 ax.bar(x+.2,[x['rescue_NLL_improvement']for x in layers],width=.4,label='Restore this block: NLL improvement')
 ax.axhline(0,color='black',lw=.7);ax.set(xlabel='LFM3 block',ylabel='NLL change per target',title='Fixed-draw training interventions; not a heldout quality result');ax.legend();fig.savefig(dest/'causal_layers.png',dpi=180);fig.savefig(dest/'causal_layers.svg');plt.close(fig)
 with(dest/'conditions.csv').open('w')as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 out={'status':'all55_training_interventions_independently_rescored','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'rows':rows,'layers':layers,'scope':p['scope']}
 (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'baseline_NLL':lookup['BF16']['NLL'],'all_K1_NLL':lookup['all_K1']['NLL']}))
if __name__=='__main__':main()
