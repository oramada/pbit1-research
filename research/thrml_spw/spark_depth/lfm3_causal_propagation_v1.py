"""Validate directionality and map downstream responses to isolated block noise."""
import json
import numpy as np
from .prepare_v1 import OUT,sha


def isolation(root,index):
 with np.load(root/f'isolated_layer{index:02d}'/'channels.npz')as f:error=f['error_SSE'].copy();signal=f['signal_SSE'].copy()
 assert error.shape==signal.shape==(24,2048)
 assert np.count_nonzero(error[:index])==0,'A supposedly isolated intervention changed an earlier block'
 assert error[index].sum()>0
 return error.sum(1),signal.sum(1)


def main():
 root=OUT/'lfm3_causal_depth_v2';result=json.loads((root/'results.json').read_text());response=[];rows=[]
 for injection in range(24):
  e,s=isolation(root,injection);response.append(np.sqrt(e/s))
  rows.append({'injected_layer':injection,'injected_block_error_power':float(e[injection]),'final_block_error_power':float(e[-1]),'final_to_injected_absolute_error_power':float(e[-1]/e[injection]),
    'final_relative_RMS_error':float(np.sqrt(e[-1]/s[-1]))})
 response=np.array(response)
 dest=OUT/'lfm3_causal_propagation_v1';dest.mkdir(exist_ok=True)
 with(dest/'responses.npz').open('wb')as f:np.savez_compressed(f,relative_RMS=response)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,6),layout='constrained');heat=np.ma.masked_where(np.tril(np.ones((24,24)),-1).astype(bool),np.log10(np.maximum(response,1e-12)))
 image=ax.imshow(heat,aspect='auto',origin='upper',cmap='viridis');ax.set(xlabel='Observed output block',ylabel='Only this block has stochastic weights',title='Downstream response to isolated LFM3 block noise (one fixed draw)');fig.colorbar(image,ax=ax,label='log10(relative RMS error)')
 fig.savefig(dest/'propagation_matrix.png',dpi=180);fig.savefig(dest/'propagation_matrix.svg');plt.close(fig)
 output={'status':'all24_isolated_interventions_verified_no_upstream_change','source_sha256':sha(__file__),'causal_result_sha256':sha(root/'results.json'),'rows':rows,
  'scope':'One fixed weight draw on training inputs. Response includes nonlinear propagation and BF16 rounding, not a Jacobian or universal Lipschitz bound.'}
 (dest/'results.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'status':output['status'],'maximum_final_error_gain':max(x['final_to_injected_absolute_error_power']for x in rows)}))
if __name__=='__main__':main()
