"""Causal three-layer CNN with native integer output-counter observers."""
from functools import lru_cache
import numpy as np
import jax.numpy as jnp
from results.thrml_spw.integer_activation_native.gated_layer import IntegerGateLayer,activation_codes,readout
from results.thrml_spw.convolution_native.operator import patches,decode

@lru_cache(None)
def gate(ni,no,rows):return IntegerGateLayer(ni,no,rows)

def encode(model):
    enc=[]
    for prefix in ['conv1','conv2','head']:
        w=model[prefix+'_weight'].astype(float);flat=w.reshape(w.shape[0],-1);lo=flat.min(1);hi=flat.max(1);c=(lo+hi)/2;r=(hi-lo)/2
        prob=np.divide(flat-c[:,None],r[:,None],out=np.zeros_like(flat),where=r[:,None]>0)/2+.5
        eps=1/(1+np.exp(8));q=np.clip(prob,eps,1-eps);command=np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8)
        field=(-4+command.astype(float)*8/255).astype(np.float32).T.copy()
        enc.append({'center':c,'radius':r,'fields':field,'commands':command,'bias':model[prefix+'_bias'].astype(float)})
    return enc

def pool(x):
    c,n,k,h,w=x.shape
    return x[:,:,:,:2*(h//2),:2*(w//2)].reshape(c,n,k,h//2,2,w//2,2).max((4,6))

def forward(x,model,enc,t,keys):
    """x[chain,image,channel,row,col]; one weight trajectory per chain/layer."""
    h=np.asarray(x,float);chains,batch=h.shape[:2];saved={}
    for li,e in enumerate(enc):
        lower=float(model['activation_lower'][li]);span=float(model['activation_span'][li]);ni,no=e['fields'].shape
        if li<2:
            prepared=[patches(h[ch],(3,3),1,1,1,lower,span) for ch in range(chains)]
            a=np.stack([v[0] for v in prepared]);codes=np.stack([v[1] for v in prepared]);mask=np.stack([v[2] for v in prepared])
        else:
            h=h.reshape(chains,batch,ni);codes=activation_codes(h,lower,span);a=codes;mask=None
        fields=np.broadcast_to(e['fields'].ravel(),(chains,ni*no)).copy()
        g,col=map(np.asarray,gate(ni,no,a.shape[1]).kernel(t)(keys[li],jnp.asarray(fields),jnp.asarray(a)))
        saved[f'gate_{li}']=g;saved[f'column_{li}']=col;saved[f'codes_{li}']=codes
        if li<2:
            saved[f'mask_{li}']=mask
            y=np.concatenate([decode(g[ch:ch+1],prepared[ch][3],codes[ch],mask[ch],t,lower,span,e['center'],e['radius'],e['bias']) for ch in range(chains)],axis=0)
            h=pool(np.maximum(y,0))
        else:
            qx=lower+span*codes.astype(float)/255
            sampled=readout(g,col[:,None,:],t,lower,span,0)
            h=qx.sum(-1)[:,:,None]*(e['center']-e['radius'])+2*e['radius']*sampled+e['bias']
        saved[f'output_{li}']=h
    return saved
