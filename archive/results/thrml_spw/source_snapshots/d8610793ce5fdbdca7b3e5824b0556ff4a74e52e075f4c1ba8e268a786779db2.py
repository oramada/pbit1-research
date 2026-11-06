"""Public-state recovery with independent patch construction and integer sums.

The F64 readout operation order is the declared native operator convention.
Alternative spatial-contraction accuracy was checked by the earlier preflight;
this path preserves byte-exact canonical quantization boundaries for receipts.
"""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,BlockGibbsSpec,SpinNode,FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional

@lru_cache(None)
def public_draw(n,t):
    block=Block([SpinNode() for _ in range(n)]);spec=BlockGibbsSpec([block],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)})
    def run(key,fields):
        program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([block],fields)],[])
        return sample_states(key,program,SamplingSchedule(1,t,1),[jnp.zeros(n,jnp.bool_)],[],[block])[0]
    return jax.jit(jax.vmap(run))

def independent_parameters(model,prefix):
    w=model[prefix+'_weight'].astype(float);no=len(w);ni=w.size//no
    centers=[];radii=[];commands=[]
    for out in range(no):
        flat=w[out].ravel();c=(max(flat)+min(flat))/2;r=(max(flat)-min(flat))/2
        centers.append(c);radii.append(r)
        probability=(flat-c)/r/2+.5 if r else np.full(ni,.5)
        e=1/(1+np.exp(8));q=np.minimum(1-e,np.maximum(e,probability));commands.append(np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8))
    fields=(-4+np.asarray(commands).astype(float)*8/255).astype(np.float32).T.copy()
    return np.asarray(centers),np.asarray(radii),fields

def im2col_independent(x,span):
    n,channels,h,w=x.shape;codes=np.zeros((n*h*w,channels*9),np.uint8);mask=np.zeros_like(codes)
    quant=np.rint(np.clip(x/span,0,1)*255).astype(np.uint8)
    for image in range(n):
        for rr in range(h):
            for cc in range(w):
                row=(image*h+rr)*w+cc
                for channel in range(channels):
                    for kr in range(3):
                        for kc in range(3):
                            ir,ic=rr+kr-1,cc+kc-1
                            if 0<=ir<h and 0<=ic<w:
                                col=channel*9+kr*3+kc;codes[row,col]=quant[image,channel,ir,ic];mask[row,col]=1
    return codes,mask

def pooling(x):
    out=np.empty((*x.shape[:3],x.shape[-2]//2,x.shape[-1]//2),float)
    for row in range(out.shape[-2]):
        for col in range(out.shape[-1]):out[:,:,:,row,col]=np.maximum(x[:,:,:,2*row:2*row+2,2*col:2*col+2],0).max((-2,-1))
    return out

def recover(x,model,t,keys):
    h=np.asarray(x,float);chains,batch=h.shape[:2];saved={}
    for li,prefix in enumerate(['conv1','conv2','head']):
        assert float(model['activation_lower'][li])==0.,'Current CNN protocol uses nonnegative post-ReLU activations'
        span=float(model['activation_span'][li]);c,r,fields=independent_parameters(model,prefix);ni,no=fields.shape
        raw=np.asarray(public_draw(ni*no,t)(keys[li],jnp.asarray(np.broadcast_to(fields.ravel(),(chains,ni*no)).copy())))
        wc=raw.sum(1,dtype=np.uint32).reshape(chains,ni,no);columns=wc.sum(1,dtype=np.uint32).astype(np.int32);saved[f'column_{li}']=columns
        if li<2:
            height,width=h.shape[-2:];parts=[im2col_independent(h[ch],span) for ch in range(chains)]
            codes=np.stack([v[0] for v in parts]);mask=np.stack([v[1] for v in parts]);g=np.einsum('cpi,cio->cpo',codes.astype(np.uint32),wc,dtype=np.uint32,optimize=False);m=np.einsum('cpi,cio->cpo',mask.astype(np.uint32),wc,dtype=np.uint32,optimize=False)
            gate=np.concatenate([g,m],axis=1).astype(np.int32);saved[f'gate_{li}']=gate;saved[f'codes_{li}']=codes;saved[f'mask_{li}']=mask
            ys=[]
            for ch in range(chains):
                # Explicit canonical readout, with exact integer sums independently recovered.
                sum_x=np.sum(mask[ch]*(0.+span*codes[ch].astype(float)/255),axis=1)
                sampled=0.*m[ch].astype(float)/t+(span/255)*g[ch].astype(float)/t
                y=sum_x[:,None]*(c-r)+2*r*sampled+model[prefix+'_bias'].astype(float)
                ys.append(y.reshape(batch,height,width,no).transpose(0,3,1,2))
            h=pooling(np.stack(ys))
        else:
            h=h.reshape(chains,batch,ni);codes=np.rint(255*np.clip(h/span,0,1)).astype(np.uint8);g=np.einsum('cbi,cio->cbo',codes.astype(np.uint32),wc,dtype=np.uint32,optimize=False).astype(np.int32);saved[f'gate_{li}']=g;saved[f'codes_{li}']=codes
            qx=0.+span*codes.astype(float)/255;sampled=0.*columns[:,None,:].astype(float)/t+(span/255)*g.astype(float)/t
            h=qx.sum(-1)[:,:,None]*(c-r)+2*r*sampled+model[prefix+'_bias'].astype(float)
        saved[f'output_{li}']=h
    return saved
