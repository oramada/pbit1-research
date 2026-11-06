"""Bounded count-only production-width timing before any high-T model run."""
from pathlib import Path
import json,time
import numpy as np
import jax
from .parameter_counts_v2 import ParameterCounts
from .parameter_program_v1 import encode
from .r1_io_v1 import sha,array_sha


def main():
    assert jax.default_backend()=='gpu'
    out=Path('results/thrml_spw/broad_verification_20261105/highT_preparation_v1/tile_benchmark_gpu');out.mkdir(parents=True,exist_ok=False)
    p=encode(np.linspace(-1,1,65536,dtype=np.float32));bank=ParameterCounts(65536);records=[]
    for T in [8192,262144]:
        begin=time.perf_counter();warm=bank.draw(p,1,T,202611791,0);cold=time.perf_counter()-begin
        path=out/f'T{T}_cold_counts.npz';np.savez_compressed(path,counts=warm['counts'].astype(np.uint32))
        runs=[]
        for index in range(3):
            begin=time.perf_counter();r=bank.draw(p,1,T,202611792+index,0);elapsed=time.perf_counter()-begin
            assert r['counts'].dtype==np.int32 and r['counts'].shape==(1,65536)
            assert np.all((r['counts']>=0)&(r['counts']<=T))
            path=out/f'T{T}_seed{202611792+index}_counts.npz';np.savez_compressed(path,counts=r['counts'].astype(np.uint32))
            runs.append({'seed':202611792+index,'seconds':elapsed,'counts_file_sha256':sha(path),'counts_array_sha256':array_sha(r['counts']),
                'scheduled_retained_node_evaluations':65536*T,'retained_node_evaluations_per_second':65536*T/elapsed})
        records.append({'samples':T,'cold_seconds_including_JIT':cold,'warm_runs':runs,'median_warm_seconds':float(np.median([r['seconds']for r in runs]))})
        (out/'progress.json').write_text(json.dumps({'last_samples':T,'records':records},indent=2)+'\n')
    result={'status':'complete_production_width_highT_count_timing_fixture','source_sha256':sha(__file__),
        'devices':list(map(str,jax.devices())),'tile_size':65536,'records':records,
        'scope':'Official THRML count-only single-tile simulator timing including host readout/decoding in draw, warm disk serialization excluded. Cold/JIT reported separately. Does not measure a full-model runtime, neural quality, physical switching rate, or hardware energy/latency.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
