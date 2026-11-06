"""Official THRML R1 worker. Only finite programs are supplied, never weights."""
import json,sys,time
from pathlib import Path
import numpy as np
import jax
from .r1_io_v1 import program,sha,array_sha
from .parameter_counts_v2 import ParameterCounts


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA officialTHRML sampler required')
    bank=ParameterCounts(65536)
    print(json.dumps({'ready':True,'backend':jax.default_backend(),'devices':list(map(str,jax.devices()))}),flush=True)
    for line in sys.stdin:
        q=json.loads(line)
        if q.get('stop'):break
        begin=time.monotonic()
        assert set(q)=={'program','program_sha256','bits','samples','seed','index','counts_output','decoded_output'}
        assert sha(q['program'])==q['program_sha256']
        assert q['bits'] in (1,4,8) and q['samples'] in (1,128,256,512,1024,2048,4096,8192)
        p=program(q['program']);r=bank.draw(p,q['bits'],q['samples'],q['seed'],q['index'])
        counts=Path(q['counts_output']);decoded=Path(q['decoded_output'])
        with counts.open('xb') as f:
            np.savez_compressed(f,counts=r['counts'].astype(np.uint16),bits=np.asarray(q['bits'],np.int64),
                samples=np.asarray(q['samples'],np.int64),seed=np.asarray(q['seed'],np.int64),index=np.asarray(q['index'],np.int64))
        value=r['decoded'].astype(np.float32)
        with decoded.open('wb') as f:np.save(f,value)
        print(json.dumps({'ok':True,'program_sha256':q['program_sha256'],'bits':q['bits'],
            'samples':q['samples'],'seed':q['seed'],'index':q['index'],'counts_sha256':sha(counts),
            'decoded_FP32_sha256':array_sha(value),'reference_weights_supplied':False,
            'finite':bool(np.isfinite(value).all()),'resources':r['resources'],'readout':r['readout'],
            'elapsed_seconds':time.monotonic()-begin}),flush=True)

if __name__=='__main__':main()
