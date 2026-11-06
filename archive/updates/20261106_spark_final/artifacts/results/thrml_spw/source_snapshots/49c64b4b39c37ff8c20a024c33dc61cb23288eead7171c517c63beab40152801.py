"""Finite-control scalar moments for full LFM3 row versus block128 ranges.

Analytical marginal moments, not neural quality or correlated device predictions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'range_moments';out.mkdir(exist_ok=True)
    inventory=json.loads((root/'meta_inventory.json').read_text())
    protocol={'source_sha256':sha(Path(__file__)),'inventory_sha256':sha(root/'meta_inventory.json'),
        'checkpoint_sha256':json.loads((root/'preparation_inventory.json').read_text())['checkpoint_files']['model.safetensors'],
        'methods':['row','block128'],'field_bits':8,'field_bounds':[-4,4],
        'scope':'Exact nominal marginal scalar moments after finite control quantization; no neural accuracy or empirical hardware claim. Tied embedding/head counted once.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    totals={m:{'weights':0,'groups':0,'bias_squared_sum':0.,'single_sample_variance_sum':0.} for m in protocol['methods']};rows=[]
    with safe_open('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors',framework='pt') as f:
        for tensor in inventory['parameters']:
            name=tensor['name'];shape=tensor['shape'];isvector=len(shape)==1
            # Group along fan-in for matrices/kernels; along vector for norm weights.
            width=int(np.prod(shape[1:])) if not isvector else shape[0]
            nrows=shape[0] if not isvector else 1
            layer={m:{'weights':0,'groups':0,'bias_squared_sum':0.,'single_sample_variance_sum':0.} for m in totals}
            for first in range(0,nrows,256):
                if isvector:w=f.get_tensor(name).float().numpy().astype(np.float64).reshape(1,-1)
                else:w=f.get_slice(name)[first:min(first+256,nrows)].float().numpy().astype(np.float64).reshape(-1,width)
                for method in totals:
                    k=width if method=='row' else min(128,width)
                    assert width%k==0
                    grouped=w.reshape(w.shape[0],width//k,k)
                    lo=grouped.min(2,keepdims=True);span=np.ptp(grouped,axis=2,keepdims=True)
                    p=np.divide(grouped-lo,span,out=np.full_like(grouped,.5),where=span!=0).clip(1e-12,1-1e-12)
                    h=np.clip((np.log(p)-np.log1p(-p))/2,-4,4)
                    codes=np.rint((h+4)*255/8).clip(0,255).astype(np.uint8)
                    decoded=codes.astype(np.float32)*np.float32(8/255)-np.float32(4)
                    q=np.exp(-np.logaddexp(0,-2*decoded.astype(np.float64)))
                    bias=lo+span*q-grouped
                    values={'weights':w.size,'groups':int(lo.size),'bias_squared_sum':float(np.sum(bias*bias)),
                        'single_sample_variance_sum':float(np.sum(span*span*q*(1-q)))}
                    for key,value in values.items():layer[method][key]+=value;totals[method][key]+=value
            rows.append({'name':name,'shape':shape,'moments':layer})
    for method,a in totals.items():
        assert a['weights']==1728801536
        a['mean_squared_bias']=a['bias_squared_sum']/a['weights']
        a['mean_single_sample_variance']=a['single_sample_variance_sum']/a['weights']
        a['scalar_MSE_at_T']={str(t):a['mean_squared_bias']+a['mean_single_sample_variance']/t for t in [16,64,256,1024,4096]}
        a['FP32_range_metadata_bytes']=8*a['groups']
    result={'summary':totals,'layers':rows,'scope':protocol['scope'],
        'block128_to_row_variance_ratio':totals['block128']['mean_single_sample_variance']/totals['row']['mean_single_sample_variance']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':totals,'variance_ratio':result['block128_to_row_variance_ratio']}),flush=True)


if __name__=='__main__':main()
