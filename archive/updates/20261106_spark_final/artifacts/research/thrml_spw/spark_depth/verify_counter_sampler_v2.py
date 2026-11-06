"""Check legacy compatibility and high-T overflow using independent raw counts."""
import json,hashlib
import numpy as np
from .prepare_v1 import OUT,sha
from . import causal_v1 as legacy
from .counter_sampler_v2 import sample_parameter
from research.thrml_spw.broad.parameter_program_v1 import encode,probabilities,decode_probability

def main():
    import torch
    seed=2026111301;index=3;w=torch.from_numpy(np.linspace(0,1,128,dtype=np.float32));program=encode(w.numpy());results=[]
    legacy.SEED=seed
    for T in [8192,32768,65536,131072,262144]:
        actual,r=sample_parameter(w,index,'storage_flat',128,T,torch,seed)
        rng=np.random.default_rng(np.random.SeedSequence([seed,index]));counts=rng.binomial(T,probabilities(program['codes']))
        expected=torch.from_numpy(decode_probability(program,counts/T).astype(np.float32)).to(torch.bfloat16)
        assert torch.equal(actual,expected)
        if T<=65535:
            old,oldr=legacy.sample_parameter(w,index,'storage_flat',128,T,torch);assert torch.equal(old,actual);assert oldr['counts_uint16_sha256']==r['counts_sha256']
        else:
            assert r['counter_dtype']=='uint32'and r['max_count']==int(counts.max())
            if T>65536:assert r['max_count']>65535
        results.append({'samples':T,'counter_dtype':r['counter_dtype'],'max_count':r['max_count'],'passed':True})
    dest=OUT/'counter_sampler_v2_verification.json';dest.write_text(json.dumps({'status':'five_budgets_verified_against_independent_decode_and_legacy_low_T','source_sha256':sha(__file__),'sampler_sha256':sha(__import__('pathlib').Path(__file__).with_name('counter_sampler_v2.py')),'checks':results},indent=2)+'\n');print(json.dumps(results))

if __name__=='__main__':main()
