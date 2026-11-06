"""NumPy-only finite all-free affine controls and synthetic device profiles."""
import numpy as np

def export_controls(path,weight,bias):
    w=np.asarray(weight,dtype=np.float32);block=128
    lower=np.stack([w[i:i+block].min(0) for i in range(0,len(w),block)])
    span=np.stack([w[i:i+block].max(0) for i in range(0,len(w),block)])-lower
    lo=np.repeat(lower,block,axis=0)[:len(w)];sp=np.repeat(span,block,axis=0)[:len(w)]
    p=np.zeros_like(w);np.divide(w-lo,sp,out=p,where=sp!=0)
    eps=1/(1+np.exp(8));p=np.clip(p.astype(float),eps,1-eps)
    field=((np.log(p)-np.log1p(-p))/2).astype(np.float32)
    codes=np.rint((np.clip(field.astype(float),-4,4)+4)*255/8).astype(np.uint8)
    np.savez(path,codes=codes,lower=lower,span=span,bias=np.asarray(bias,np.float32))


def decode(code):return (-4+np.asarray(code,dtype=float)*8/255).astype(np.float32)


def device(stage,seed,size):
    z=np.random.default_rng(np.random.SeedSequence([2026114600,stage,seed])).standard_normal((2,size))
    return np.exp(.2*z[1]-.02),.2*z[0]


