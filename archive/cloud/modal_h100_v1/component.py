"""Small official-sampler native H100 check before full-model replay."""
import os
import json
import subprocess
import sys
from pathlib import Path


def main():
    import torch
    assert 'H100' in torch.cuda.get_device_name(0)
    output=Path('h100_component_v1');output.mkdir(exist_ok=False)
    script="""import json,numpy as np,jax
from research.thrml_spw.native import NativeProbabilityBank
b=NativeProbabilityBank(np.array([0.,.125,.5,.875,1.]))
s=b.sample(202610703,8192,2)
assert s.shape==(2,8192,5) and not s[:,:,0].any() and s[:,:,-1].all()
print(json.dumps({'backend':list(map(str,jax.devices())),'logical_pbits':b.logical_pbits,'samples':8192,'shape':list(s.shape),'sample_means':s.mean((0,1)).tolist()}))"""
    with (output/'stderr.log').open('x') as log:
        run=subprocess.run([sys.executable,'-m','research.thrml_spw.gpu_environment','--','/opt/thrml/bin/python','-c',script],stdout=subprocess.PIPE,stderr=log,text=True,check=True)
    result={'status':'H100_official_native_component_passed','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'sampler':json.loads(run.stdout),'scope':'H100 native component only, not full model or BF16 equivalence.'}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
