"""Independent endpoint-law audit of full finite-control range moments."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'range_moments'
    result=json.loads((out/'results.json').read_text())
    inventory=json.loads((root/'meta_inventory.json').read_text())
    byname={r['name']:r for r in result['layers']}
    assert set(byname)=={p['name'] for p in inventory['parameters']}
    h=np.arange(256,dtype=np.float32)*np.float32(8/255)-np.float32(4)
    table=1/(1+np.exp(-2*h.astype(np.float64)))
    sums={method:{'weights':0,'groups':0,'bias_squared_sum':0.,'single_sample_variance_sum':0.} for method in ['row','block128']}
    checks=0
    with safe_open('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors',framework='pt') as f:
        for tensor in inventory['parameters']:
            name=tensor['name'];shape=tensor['shape'];vector=len(shape)==1
            width=int(np.prod(shape[1:])) if not vector else shape[0]
            n=shape[0] if not vector else 1
            totals={m:{k:0 for k in sums[m]} for m in sums}
            for first in range(0,n,128):
                w=(f.get_tensor(name) if vector else f.get_slice(name)[first:first+128]).float().numpy().astype(np.float64).reshape(-1,width)
                for method in totals:
                    size=width if method=='row' else min(128,width)
                    a=w.reshape(-1,size);low=np.min(a,axis=1)[:,None];high=np.max(a,axis=1)[:,None];span=high-low
                    requested=np.divide(a-low,span,out=np.full_like(a,.5),where=span!=0).clip(1e-12,1-1e-12)
                    field=(np.log(requested)-np.log1p(-requested))/2
                    codes=np.rint((field.clip(-4,4)+4)*(255/8)).clip(0,255).astype(np.uint8)
                    probability=table[codes]
                    mean=(1-probability)*low+probability*high
                    # Two endpoint distribution, independent of p*(1-p) variance formula.
                    variance=(1-probability)*(low-mean)**2+probability*(high-mean)**2
                    d={'weights':a.size,'groups':a.shape[0],'bias_squared_sum':float(np.sum((mean-a)**2)),
                       'single_sample_variance_sum':float(np.sum(variance))}
                    for k,v in d.items():totals[method][k]+=v;sums[method][k]+=v
            for method,values in totals.items():
                expected=byname[name]['moments'][method]
                for k,v in values.items():
                    np.testing.assert_allclose(v,expected[k],rtol=2e-11,atol=2e-12);checks+=1
    for method,values in sums.items():
        expected=result['summary'][method]
        for k,v in values.items():np.testing.assert_allclose(v,expected[k],rtol=2e-11,atol=2e-10)
        assert values['weights']==1728801536
        for t in [16,64,256,1024,4096]:
            value=(values['bias_squared_sum']+values['single_sample_variance_sum']/t)/values['weights']
            np.testing.assert_allclose(value,expected['scalar_MSE_at_T'][str(t)],rtol=2e-11,atol=2e-14)
        assert expected['FP32_range_metadata_bytes']==8*values['groups']
    audit={'source_sha256':sha(Path(__file__)),'results_sha256':sha(out/'results.json'),
        'status':'all_tensor_moments_verified_by_endpoint_law','tensor_metric_checks':checks,
        'unique_weights_per_method':1728801536,'methods':2,
        'scope':'Independent full-coordinate nominal marginal-moment audit, different row chunks and endpoint probability law. No task-quality or empirical hardware conclusion.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
