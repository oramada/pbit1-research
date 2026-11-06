"""Finite all-free controls with explicit nested local range granularity."""
import numpy as np

def build(weight,bias,group):
    w=np.asarray(weight,np.float32)
    if group not in (16,32,64,128) or w.ndim!=2 or len(w)%group:raise ValueError('Nested, full input groups required')
    view=w.reshape(-1,group,w.shape[1]);lower=view.min(1);upper=view.max(1);span=upper-lower;lo=np.repeat(lower,group,0);sp=np.repeat(span,group,0);p=np.zeros_like(w);np.divide(w-lo,sp,out=p,where=sp!=0);eps=1/(1+np.exp(8));p=np.clip(p.astype(float),eps,1-eps);field=((np.log(p)-np.log1p(-p))/2).astype(np.float32);codes=np.rint((np.clip(field.astype(float),-4,4)+4)*255/8).astype(np.uint8)
    return {'codes':codes,'lower':lower,'span':span,'bias':np.asarray(bias,np.float32),'group_size':np.array(group,np.uint16)}

def expand(a,group):
    return np.repeat(a['lower'],group,0),np.repeat(a['span'],group,0)

def decode(code):return (-4+np.asarray(code,dtype=float)*8/255).astype(np.float32)

def ideal_variance(weight,group):
    w=np.asarray(weight,np.float64);view=w.reshape(-1,group,w.shape[1]);lower=view.min(1);upper=view.max(1)
    return (w-np.repeat(lower,group,0))*(np.repeat(upper,group,0)-w)
