"""Read-only independent score and layer/channel summaries for LFM3 cohort."""
import json,math
from pathlib import Path
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 root=OUT/'lfm3_training_depth_v1';protocol=json.loads((root/'protocol.json').read_text());r=json.loads((root/'results.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
 dest=OUT/'lfm3_training_depth_analysis_v1';dest.mkdir(exist_ok=True)
 metrics=[];signals=[];errors=[]
 for case in protocol['cases']:
  folder=root/case['name'];loss=[]
  for row in range(32):
   with np.load(folder/f'row{row:02d}.npz')as f:
    assert f['token_NLL'].shape==(256,) and f['input_ids'].shape==(257,) and np.isfinite(f['token_NLL']).all();loss.extend(f['token_NLL'])
  mean=float(np.mean(loss));summary=json.loads((folder/'results.json').read_text());assert mean==summary['NLL']and len(loss)==summary['targets']==8192
  with np.load(folder/'channels.npz')as f:
   s=f['signal_SSE'];e=f['error_SSE'];assert s.shape==e.shape==(24,2048)and(s>0).all()and(e>=0).all()
   np.testing.assert_allclose(np.sqrt(e.sum(1)/s.sum(1)),summary['layer_relative_RMSE'],rtol=1e-14)
   signals.append(s.copy());errors.append(e.copy())
  metrics.append({'name':case['name'],'NLL':mean,'PPL':math.exp(mean)})
 delta=np.array([x['NLL']for x in metrics[2:]])-metrics[0]['NLL'];half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
 np.testing.assert_allclose(np.exp(delta.mean()),r['PPL_ratio'],rtol=1e-14)
 np.testing.assert_allclose(np.exp([delta.mean()-half,delta.mean()+half]),r['MC95_t4_PPL_ratio_interval'],rtol=1e-14)
 assert np.count_nonzero(errors[0])==0
 avg_e=np.mean(errors[2:],axis=0);s=signals[0];layer_rrmse=np.sqrt(avg_e.sum(1)/s.sum(1));types=r['cases'][0]['layer_types'];layers=[]
 for i in range(24):
  top=np.argsort(avg_e[i])[-20:][::-1]
  layers.append({'layer':i,'type':types[i],'pooled_relative_RMSE':float(layer_rrmse[i]),'top20_channels_by_absolute_error':top.tolist(),'top20_error_fraction':float(avg_e[i,top].sum()/avg_e[i].sum()),'worst_channel':int(top[0])})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
 colors={'conv':'#478cbf','full_attention':'#d17934','sliding_attention':'#729447'}
 axes[0].bar(range(24),layer_rrmse*100,color=[colors.get(t,'#999999')for t in types]);axes[0].set(ylabel='Block output relative RMS error (%)',xlabel='LFM3 block',title='One p-bit per parameter: five-seed training-cohort diagnostic')
 image=axes[1].imshow(np.log10(np.maximum(np.sqrt(avg_e/s),1e-6)),aspect='auto',interpolation='nearest',cmap='magma');axes[1].set(ylabel='LFM3 block',xlabel='Hidden channel');fig.colorbar(image,ax=axes[1],label='log10(relative RMS error)')
 fig.savefig(dest/'layer_channel_error.png',dpi=180);fig.savefig(dest/'layer_channel_error.svg');plt.close(fig)
 result={'status':'training_scores_and_channel_aggregates_audited','results_sha256':sha(root/'results.json'),'source_sha256':sha(__file__),'metrics':metrics,'PPL_ratio':r['PPL_ratio'],'MC95_t4_PPL_ratio_interval':r['MC95_t4_PPL_ratio_interval'],'layers':layers,'scope':'Training cohort diagnostic, no heldout or causal-attribution claim. Propagated block error is not isolated layer sensitivity.'}
 (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['layers','metrics']}))
if __name__=='__main__':main()
