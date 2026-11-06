"""Cross-chunk group-7 decoding and backward compatibility fixtures."""
import json,hashlib
import numpy as np
from .prepare_v1 import OUT,sha
from .weight_survey_v1 import mapped
from .counter_sampler_v3 import sample_parameter
from . import counter_sampler_v2 as previous

def main():
    import torch
    records=[];seed=2026111801
    for shape,group,T,layout in [((37500,1,7),7,32,'storage_flat'),((37500,1,7),7,131072,'storage_flat'),((17,19),128,8192,'transposed_matrix'),((21,),7,65536,'storage_flat')]:
        n=int(np.prod(shape));i=np.arange(n);w=((i%7)/6+.1*((i//7)%5)).astype(np.float32).reshape(shape);raw=torch.from_numpy(w)
        actual,r=sample_parameter(raw,2,layout,group,T,torch,seed)
        a=w.T.copy()if layout=='transposed_matrix'else w;codes,lo,span,mean,var=mapped(a,group)
        fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4);prob=1/(1+np.exp(-2*fields.astype(np.float64)))
        rng=np.random.default_rng(np.random.SeedSequence([seed,2]));counts=rng.binomial(T,prob);dtype=np.uint16 if T<=65535 else np.uint32
        assert hashlib.sha256(counts.astype(dtype).tobytes()).hexdigest()==r['counts_sha256']
        value=(np.repeat(lo,group)[:n]+np.repeat(span,group)[:n]*counts.astype(np.float64)/T).astype(np.float32).reshape(a.shape)
        if layout=='transposed_matrix':value=value.T.copy()
        expected=torch.from_numpy(value).to(torch.bfloat16);assert torch.equal(actual,expected)
        if group==128:
            old,oldr=previous.sample_parameter(raw,2,layout,group,T,torch,seed);assert torch.equal(old,actual)and oldr['counts_sha256']==r['counts_sha256']
        records.append({'shape':list(shape),'group':group,'samples':T,'layout':layout,'exact_reference_match':True})
    out={'status':'all4_aligned_group_and_dtype_fixtures_passed','source_sha256':sha(__file__),'sampler_sha256':sha(__import__('pathlib').Path(__file__).with_name('counter_sampler_v3.py')),'checks':records}
    (OUT/'counter_sampler_v3_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
