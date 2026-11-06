"""Loss-aware evidence format: retained hashes permit verification, not decoding.

Original patch arrays are transient. Recovery must rerun the pinned public sampler
and reproduce each canonical array before its digest can be accepted.
"""
import hashlib,json
import numpy as np

NAMES=('codes_0','mask_0','gate_0','column_0','output_0',
       'codes_1','mask_1','gate_1','column_1','output_1',
       'codes_2','gate_2','column_2','output_2')

def digest(name,array):
    a=np.ascontiguousarray(array)
    if a.dtype.hasobject:raise ValueError('Object arrays are not scientific evidence')
    header=json.dumps({'name':name,'dtype':a.dtype.str,'shape':list(a.shape)},sort_keys=True,separators=(',',':')).encode()
    return np.frombuffer(hashlib.sha256(header+b'\x00'+a.tobytes(order='C')).digest(),dtype=np.uint8).copy()

def receipt(raw):
    if set(raw)!=set(NAMES):raise ValueError('Incomplete or unexpected observation fields')
    chains=raw['output_2'].shape[0]
    if not all(raw[name].shape[0]==chains for name in NAMES):raise ValueError('Inconsistent bank dimension')
    return np.stack([np.stack([digest(name,raw[name][chain]) for name in NAMES]) for chain in range(chains)])
