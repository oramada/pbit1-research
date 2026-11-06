"""Separate changes in absolute error from changes in the relative-error denominator."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 root=OUT/'lfm3_training_depth_v1';result=json.loads((root/'results.json').read_text())
 with np.load(root/'BF16/channels.npz')as f:signal=f['signal_SSE'].sum(1)/8224
 errors=[]
 for case in result['cases'][2:]:
  with np.load(root/case['case']['name']/'channels.npz')as f:errors.append(f['error_SSE'].sum(1)/8224)
 error=np.mean(errors,axis=0);relative=np.sqrt(error/signal);rows=[]
 for layer in range(24):
  item={'layer':layer,'type':result['cases'][0]['layer_types'][layer],'signal_squared_norm_per_position':float(signal[layer]),'error_squared_norm_per_position':float(error[layer]),'relative_RMS_error':float(relative[layer])}
  if layer:
   item.update({'absolute_error_power_ratio_to_previous_block':float(error[layer]/error[layer-1]),'signal_power_ratio_to_previous_block':float(signal[layer]/signal[layer-1]),'relative_RMS_error_ratio_to_previous_block':float(relative[layer]/relative[layer-1])})
  rows.append(item)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');axes[0].plot(range(24),signal,label='Signal');axes[0].plot(range(24),error,label='Error');axes[0].set(xlabel='Block',ylabel='Mean squared vector norm',yscale='log',title='Absolute signal and error power');axes[0].legend()
 axes[1].plot(range(24),relative*100,marker='.');axes[1].set(xlabel='Block',ylabel='Relative RMS error (%)',title='Relative error includes a changing denominator')
 dest=OUT/'lfm3_error_propagation_v1';dest.mkdir(exist_ok=True);fig.savefig(dest/'absolute_and_relative.png',dpi=180);fig.savefig(dest/'absolute_and_relative.svg');plt.close(fig)
 out={'status':'all24_block_error_scales_decomposed','source_sha256':sha(__file__),'training_result_sha256':sha(root/'results.json'),'rows':rows,'scope':'Descriptive propagation from five training draws. Error includes newly injected noise and upstream effects; block ratios do not isolate causal amplification or denoising.'}
 (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows[12]))
if __name__=='__main__':main()
