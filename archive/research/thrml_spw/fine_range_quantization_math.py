"""Fixed simple low-bit controls; no calibration, labels or variant selection."""
import hashlib
import numpy as np
from .block_quantization_controls import quantize
def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def uniform(weight,bits,group,method):
    w=np.asarray(weight,np.float32);view=w.reshape(-1,group,w.shape[1]);groups=view.shape[0]*view.shape[2]
    if method=='minmax':
        lo=view.min(1,keepdims=True);scale=(view.max(1,keepdims=True)-lo)/((1<<bits)-1);q=np.zeros_like(view);np.divide(view-lo,scale,out=q,where=scale!=0);codes=np.clip(np.rint(q),0,(1<<bits)-1).astype(np.uint8);result=(lo+codes.astype(np.float32)*scale).reshape(w.shape);meta_bits=64*groups
    else:
        result=quantize(w,bits,'symmetric' if method=='symmetric' else 'affine_zero',group)
        if method not in ['symmetric','affine_zero']:raise ValueError('Unknown lattice')
        meta_bits=groups*(32 if method=='symmetric' else 32+bits)
    return result.astype(np.float32),{'groups':groups,'code_bits':w.size*bits,'range_metadata_bits':meta_bits}
def mean_weight(controls,group,codebook):
    return (np.repeat(controls['lower'],group,0).astype(float)+np.repeat(controls['span'],group,0).astype(float)*codebook[controls['codes']]).astype(np.float32)
