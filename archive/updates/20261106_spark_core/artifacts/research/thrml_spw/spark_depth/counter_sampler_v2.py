"""Ideal Bernoulli-count sampler with explicit seed and nonwrapping counters."""
import hashlib,math
import numpy as np
from .weight_survey_v1 import mapped

CHUNK=262144

def sample_parameter(raw,index,layout,group,T,torch,seed):
    if type(T)is not int or not 1<=T<=2**32-1:raise ValueError('Samples must fit uint32')
    if group not in [16,32,64,128]or layout not in ['storage_flat','transposed_matrix']:raise ValueError('Unsupported grouping')
    dtype=np.uint16 if T<=65535 else np.uint32
    w=raw.float().numpy();a=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w;flat=a.reshape(-1);out=np.empty(flat.shape,np.float32)
    rng=np.random.default_rng(np.random.SeedSequence([seed,index]));h=hashlib.sha256();observed=expected=variance=0.;max_count=0
    for start in range(0,len(flat),CHUNK):
        stop=min(start+CHUNK,len(flat));codes,lo,span,mean,var=mapped(flat[start:stop],group)
        fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4);p=1/(1+np.exp(-2*fields.astype(np.float64)))
        counts=rng.binomial(T,p).astype(dtype);h.update(counts.tobytes());max_count=max(max_count,int(counts.max()))
        lower=np.repeat(lo,group)[:len(counts)].astype(np.float64);scale=np.repeat(span,group)[:len(counts)].astype(np.float64)
        value=lower+scale*counts.astype(np.float64)/T;out[start:stop]=value;error=value-mean;v=var/T
        observed+=float(error@error);expected+=float(v.sum());pq=p*(1-p);variance+=float(np.sum(2*v*v+scale**4*pq*(1-6*pq)/T**3))
    value=out.reshape(a.shape)
    if layout=='transposed_matrix'and w.ndim==2:value=value.T.copy()
    rounded=torch.from_numpy(value.copy()).to(torch.bfloat16)
    record={'counts_sha256':h.hexdigest(),'counter_dtype':np.dtype(dtype).name,'max_count':max_count,
        'BF16_sha256':hashlib.sha256(rounded.view(torch.uint16).numpy().tobytes()).hexdigest(),'parameters':len(flat),'samples':T,
        'layout':layout if w.ndim==2 else'storage_flat','group':group,'range_groups':math.ceil(len(flat)/group),
        'SSE_about_mean':observed,'expected_SSE':expected,'variance_of_SSE':variance}
    return rounded,record
