"""Explicit eight-bit control artifacts, with two-bit modes and float32 ranges.

Artifacts store no original converted FP32 weights. Decoding still expands control
fields for the software sampler, so artifact size is not runtime memory or area.
"""
import hashlib,json
from pathlib import Path
import numpy as np


def save_controls(path,weight,bias,block_size=128,bound=4.):
    w=np.asarray(weight,dtype=np.float32);bias=np.asarray(bias,dtype=np.float32)
    if w.ndim!=2 or not w.size or not np.isfinite(w).all() or bias.shape!=(w.shape[1],) or not np.isfinite(bias).all():raise ValueError('Finite matrix and matching bias required')
    if not isinstance(block_size,int) or block_size<1 or not np.isfinite(bound) or bound<=0:raise ValueError('Valid group size and finite bound required')
    lower=np.stack([w[i:i+block_size].min(0) for i in range(0,len(w),block_size)])
    span=np.stack([w[i:i+block_size].max(0) for i in range(0,len(w),block_size)])-lower
    lo=np.repeat(lower,block_size,axis=0)[:len(w)];radius=np.repeat(span,block_size,axis=0)[:len(w)]
    p=np.zeros_like(w);np.divide(w-lo,radius,out=p,where=radius!=0);p=np.clip(p,0,1).astype(np.float64).reshape(-1)
    free=(p>0)&(p<1);field=((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32).astype(np.float64)
    code=np.rint((np.clip(field,-bound,bound)+bound)*255/(2*bound)).astype(np.uint8)
    modes=np.zeros(len(p),dtype=np.uint8);modes[p==0]=1;modes[p==1]=2
    padded=np.pad(modes,(0,(-len(modes))%4)).reshape(-1,4)
    packed=np.bitwise_or.reduce(padded<<np.array([0,2,4,6],dtype=np.uint8),axis=1)
    header={'schema':'spw-control8-v1','shape':list(w.shape),'block_size':block_size,'bits':8,'bound':float(bound),'beta':1.,
        'modes':'0=free,1=clamp0,2=clamp1; low two bits encode first weight','range_clipped_fields':int(np.sum(np.abs(field)>bound)),
        'target_weight_sha256':hashlib.sha256(w.tobytes()).hexdigest(),'scope':'Eight-bit controls, endpoint modes, shared affine ranges and deterministic bias; not one total bit per weight or physical memory measurement'}
    meta=np.frombuffer(json.dumps(header,sort_keys=True).encode(),dtype=np.uint8)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,codes=code,modes=packed,lower=lower,span=span,bias=bias,header=meta)
    payload=code.nbytes+packed.nbytes+lower.nbytes+span.nbytes+bias.nbytes
    return {'path':str(path),'weights':w.size,'free_nodes':int(free.sum()),'numeric_payload_bytes':payload,'header_bytes':meta.nbytes,'archive_bytes':path.stat().st_size,
        'npz_overhead_bytes':path.stat().st_size-payload-meta.nbytes,'nominal_FP32_weight_bias_bytes':w.nbytes+bias.nbytes}


def load_controls(path):
    with np.load(path,allow_pickle=False) as source:
        if set(source.files)!={'codes','modes','lower','span','bias','header'}:raise ValueError('Unexpected control artifact entries')
        code,modes,lower,span,bias,raw=[source[k].copy() for k in ['codes','modes','lower','span','bias','header']]
    header=json.loads(raw.tobytes().decode());shape=tuple(header['shape']);block=header['block_size'];bound=header['bound']
    if header['schema']!='spw-control8-v1' or header['bits']!=8 or header['beta']!=1. or len(shape)!=2 or min(shape)<1 or block<1 or not np.isfinite(bound) or bound<=0:raise ValueError('Unsupported header')
    n=int(np.prod(shape));groups=(shape[0]+block-1)//block
    if code.dtype!=np.uint8 or code.ndim!=1 or modes.dtype!=np.uint8 or modes.shape!=((n+3)//4,):raise ValueError('Bad packed arrays')
    decoded=((modes[:,None]>>np.array([0,2,4,6],dtype=np.uint8))&3).reshape(-1)
    if np.any(decoded[:n]>2) or np.any(decoded[n:]!=0):raise ValueError('Invalid mode or padding')
    decoded=decoded[:n];free=np.flatnonzero(decoded==0)
    if len(code)!=len(free):raise ValueError('Control code count does not match modes')
    if lower.shape!=(groups,shape[1]) or span.shape!=lower.shape or bias.shape!=(shape[1],):raise ValueError('Metadata shape mismatch')
    if any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in [lower,span,bias]) or np.any(span<0):raise ValueError('Bad range/bias format')
    fields=(-bound+code.astype(np.float64)*(2*bound/255)).astype(np.float32)
    probability=np.zeros(n,dtype=np.float64);probability[decoded==2]=1
    probability[free]=np.exp(-np.logaddexp(0,-2*fields.astype(np.float64)))
    return {'header':header,'codes':code,'modes':decoded,'free_indices':free,'fields':fields,'probabilities':probability.reshape(shape),'lower':lower,'span':span,'bias':bias}


def expected_weights(loaded):
    shape=tuple(loaded['header']['shape']);block=loaded['header']['block_size']
    lower=np.repeat(loaded['lower'],block,axis=0)[:shape[0]];span=np.repeat(loaded['span'],block,axis=0)[:shape[0]]
    return lower+span*loaded['probabilities']
