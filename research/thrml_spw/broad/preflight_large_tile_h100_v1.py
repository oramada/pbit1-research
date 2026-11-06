"""Actual65536lane R1 sampler at all primary equal-work formats andT8192."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import jax
from .parameter_program_v1 import encode,expectation
from .parameter_counts_v2 import ParameterCounts

OUT=Path('results/thrml_spw/broad_verification_20261105/large_tile_h100_v1')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert jax.default_backend()=='gpu';OUT.mkdir(exist_ok=False)
    rng=np.random.default_rng(202611930)
    weights=rng.integers(-127,128,65535,dtype=np.int32).astype(np.float32)/64
    weights[:128]=.125;weights[128:256]=-.25
    program=encode(weights);sampler=ParameterCounts(65536);cases=[]
    for bits in [1,4,8]:
        samples=8192//bits;begin=time.monotonic()
        result=sampler.draw(program,bits,samples,202611931,0,replay=True)
        elapsed=time.monotonic()-begin;path=OUT/f'K{bits}.npz'
        arrays={**program,'weight':weights,'counts':result['counts'],'decoded':result['decoded'],
            'expectation':expectation(program,bits)}
        records=[]
        for index,record in enumerate(result['raw']):
            for name in ['fields','states','padded_counts']:arrays[f'raw{index}_{name}']=record[name]
            records.append({k:record[k] for k in ['tile','begin','width','bit']})
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        cases.append({'bits':bits,'samples':samples,'seed':202611931,'parameter_index':0,'path':str(path),
            'sha256':sha(path),'raw_tiles':records,'resources':result['resources'],
            'count_and_public_replay_seconds_including_JIT':elapsed,'readout':result['readout']})
        del result,arrays
    report={'status':'full65536lane_primarybudget_publicstates_counts_and_readouts_produced_pending_audit',
        'cases':cases,'backend':jax.default_backend(),'devices':list(map(str,jax.devices())),
        'sources_sha256':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('parameter_counts_v2.py'),
            Path(__file__).with_name('parameter_counts_v1.py'),Path(__file__).with_name('parameter_program_v1.py')]},
        'scope':'Synthetic65535coordinate parameter,65536lane sampler, exact1cellpadding charged, constant andshorttailgroups retained. Countpluspublicreplaytime is simulatoroverhead, not physical energy/latency or neural quality.'}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'}))

if __name__=='__main__':main()
