"""Actual official-THRML state/count evidence for flat groups and short tails."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import jax
from .parameter_program_v1 import encode,expectation,integer_control
from .parameter_counts_v1 import ParameterCounts

OUT=Path('results/thrml_spw/broad_verification_20261105/parameter_counts_cpu_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    assert jax.default_backend()=='cpu'
    shapes=[(2,129),(1,),(3,5)]
    weights=[np.linspace(-1,1,258,dtype=np.float32).reshape(2,129),
             np.asarray([.125],np.float32),np.full((3,5),-.25,np.float32)]
    sampler=ParameterCounts(64);cases=[]
    for index,(shape,w) in enumerate(zip(shapes,weights)):
        program=encode(w)
        for bits in [1,4,8]:
            # Equal retained stochastic-element work for every format.
            samples=1024//bits;seed=202611920+index
            result=sampler.draw(program,bits,samples,seed,index,replay=True)
            path=OUT/f'p{index}_K{bits}.npz'
            arrays={**program,'weight':w,'counts':result['counts'],'decoded':result['decoded'],
                'expectation':expectation(program,bits),
                **{'int'+str(k):integer_control(w,k) for k in [2,4,8]}}
            records=[]
            for ordinal,raw in enumerate(result['raw']):
                for name in ['fields','states','padded_counts']:arrays[f'raw{ordinal}_{name}']=raw[name]
                records.append({k:raw[k] for k in ['tile','begin','width','bit']})
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
            cases.append({'parameter_index':index,'shape':list(shape),'bits':bits,'samples':samples,'seed':seed,
                'path':str(path),'sha256':sha(path),'raw_tiles':records,'resources':result['resources']})
    result={'status':'official_CPU_TH RML_full_public_states_counts_and_programs_produced_pending_independent_audit'.replace('TH RML','THRML'),
        'cases':cases,'backend':jax.default_backend(),'jax_version':jax.__version__,
        'sources_sha256':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('parameter_program_v1.py'),
            Path(__file__).with_name('parameter_counts_v1.py'),Path('research/thrml_spw/native.py'),
            Path('research/thrml_spw/controlled_positive_native.py'),Path('research/thrml_spw/lfm3/counted_bank.py')]},
        'elapsed_seconds':time.monotonic()-start,'scope':'Synthetic full parameter vectors and officialpublicstate replay, no neural model or real test data.'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))

if __name__=='__main__':main()
