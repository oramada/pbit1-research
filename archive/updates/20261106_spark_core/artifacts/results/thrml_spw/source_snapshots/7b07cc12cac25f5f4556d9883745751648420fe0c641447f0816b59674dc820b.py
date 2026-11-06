"""Native public-state proof and uint32 overflow guard before high-T scoring."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import jax
from .parameter_counts_v2 import ParameterCounts
from .parameter_program_v1 import encode,probabilities
from .r1_io_v1 import sha,array_sha


def main():
    backend=jax.default_backend();assert backend in ('cpu','gpu')
    out=Path('results/thrml_spw/broad_verification_20261105/highT_preparation_v1')/('primitive_'+backend)
    out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    p=encode(np.linspace(-1,1,17,dtype=np.float32));bank=ParameterCounts(8);records=[]
    popcount=np.array([int(x).bit_count() for x in range(256)],np.uint8)
    for T in [8192,262144]:
        r=bank.draw(p,1,T,202611790,4,replay=True);counts=r['counts']
        arrays={};packed_sha=[]
        for item in r['raw']:
            states=item['states'];assert states.shape==(T,8) and states.dtype==np.bool_
            packed=np.packbits(states,axis=0,bitorder='little')
            independent=popcount[packed].sum(axis=0,dtype=np.uint64)
            np.testing.assert_array_equal(independent,item['padded_counts'])
            np.testing.assert_array_equal(independent[:item['width']],counts[item['bit'],item['begin']:item['begin']+item['width']])
            arrays[f'tile{item["tile"]}_packed']=packed
            packed_sha.append(array_sha(packed))
        raw=out/f'T{T}_public_states.npz';np.savez_compressed(raw,**arrays)
        path=out/f'T{T}_counts.npz';np.savez_compressed(path,counts=counts.astype(np.uint32))
        with np.load(path,allow_pickle=False)as saved:
            assert saved['counts'].dtype==np.uint32;np.testing.assert_array_equal(saved['counts'],counts)
        lo=np.repeat(p['lower'],128)[:17].astype(np.float64);span=np.repeat(p['span'],128)[:17].astype(np.float64)
        independent_decode=(lo+span*(counts[0].astype(np.float64)/T)).astype(np.float32)
        np.testing.assert_array_equal(independent_decode,r['decoded'].astype(np.float32))
        overflow_would_corrupt=bool(np.any(counts.astype(np.uint16).astype(np.int64)!=counts))
        assert overflow_would_corrupt==(T==262144)
        assert r['resources']['logical_retained_device_evaluations']==17*T
        assert r['resources']['scheduled_retained_device_evaluations']==24*T
        records.append({'samples':T,'public_bits_checked':24*T,'packed_state_SHA256':packed_sha,
            'counts_SHA256':array_sha(counts),'decoded_FP32_SHA256':array_sha(independent_decode),
            'raw_public_file':str(raw),'raw_public_file_sha256':sha(raw),'count_file':str(path),'count_file_sha256':sha(path),
            'max_count':int(counts.max()),'uint16_would_corrupt_counts':overflow_would_corrupt,'saved_dtype':'uint32',
            'resources':r['resources']})
    result={'status':'highT_native_public_counts_uint32_roundtrip_and_independent_FP32_readout_verified',
        'backend':backend,'devices':list(map(str,jax.devices())),'source_sha256':sha(__file__),
        'seed':202611790,'parameter_index':4,'tile_size':8,'records':records,
        'elapsed_seconds':time.monotonic()-begin,
        'scope':'Primitive high-sample-count fixture only. Public states and counts use official THRML; no neural task score or BF16-quality conclusion. Production has65536lanes; previously audited at8192samples, while this fixture checks262144samples with8lanes.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
