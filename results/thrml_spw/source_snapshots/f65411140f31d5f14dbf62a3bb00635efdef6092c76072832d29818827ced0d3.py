"""Finite field ranges, logical code precision and optional endpoint modes."""
import numpy as np

ENCODINGS=[
    {'name':'clamped_B4_b8','bound':4.,'bits':8,'clamp_endpoints':True},
    {'name':'free_B4_b8','bound':4.,'bits':8,'clamp_endpoints':False},
    {'name':'free_B8_b8','bound':8.,'bits':8,'clamp_endpoints':False},
    {'name':'free_B8_b10','bound':8.,'bits':10,'clamp_endpoints':False},
]


def validate(bits,bound):
    if not isinstance(bits,int) or not 2<=bits<=16 or not np.isfinite(bound) or not 0<bound<=16:
        raise ValueError('Supported finite controller:2..16 bits and bound in(0,16]')


def quantize(requested,bits,bound):
    validate(bits,bound);requested=np.asarray(requested,dtype=np.float64)
    if not np.isfinite(requested).all():raise ValueError('Finite requested fields required')
    levels=(1<<bits)-1;dtype=np.uint8 if bits<=8 else np.uint16
    return np.rint((np.clip(requested,-bound,bound)+bound)*levels/(2*bound)).astype(dtype)


def decode(codes,bits,bound):
    validate(bits,bound);codes=np.asarray(codes);levels=(1<<bits)-1
    if codes.dtype.kind not in 'ui' or np.any(codes<0) or np.any(codes>levels):raise ValueError('Valid integer control codes required')
    return (-bound+codes.astype(np.float64)*(2*bound/levels)).astype(np.float32)


def encoded_weights(weights,encoding):
    validate(encoding['bits'],encoding['bound']);result=[]
    epsilon=1/(1+np.exp(2*encoding['bound']))
    for weight in weights:
        w=np.asarray(weight,dtype=np.float64)
        if not np.isfinite(w).all() or np.any((w<0)|(w>1)):raise ValueError('Weights must be in[0,1]')
        result.append(w.copy() if encoding['clamp_endpoints'] else np.clip(w,epsilon,1-epsilon))
    return result
