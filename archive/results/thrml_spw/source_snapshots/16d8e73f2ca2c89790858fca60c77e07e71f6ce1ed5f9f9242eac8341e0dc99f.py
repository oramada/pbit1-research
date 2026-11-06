"""Vectorized patch preparation with unchanged native integer observation."""
import numpy as np
import jax.numpy as jnp
from results.thrml_spw.fashion_cnn_native.inference.native import gate,pool
from results.thrml_spw.integer_activation_native.gated_layer import activation_codes,readout
from results.thrml_spw.convolution_native.operator import decode

def vector_patches(x,span):
    n,channels,h,w=x.shape
    quant=activation_codes(x,0.,span)
    codepad=np.pad(quant,((0,0),(0,0),(1,1),(1,1)),constant_values=0)
    validpad=np.pad(np.ones_like(quant,np.uint8),((0,0),(0,0),(1,1),(1,1)),constant_values=0)
    def matrix(a):
        v=np.lib.stride_tricks.sliding_window_view(a,(3,3),axis=(-2,-1))
        return np.ascontiguousarray(v.transpose(0,2,3,1,4,5).reshape(n*h*w,channels*9))
    codes=matrix(codepad);mask=matrix(validpad)
    return np.concatenate([codes,mask]),codes,mask,(n,h,w)

def forward(x,model,enc,t,keys):
    h=np.asarray(x,float);chains,batch=h.shape[:2];saved={}
    for li,e in enumerate(enc):
        lower=float(model['activation_lower'][li]);span=float(model['activation_span'][li]);assert lower==0.
        ni,no=e['fields'].shape
        if li<2:
            prepared=[vector_patches(h[ch],span) for ch in range(chains)]
            a=np.stack([v[0] for v in prepared]);codes=np.stack([v[1] for v in prepared]);mask=np.stack([v[2] for v in prepared])
        else:
            h=h.reshape(chains,batch,ni);codes=activation_codes(h,0.,span);a=codes
        fields=np.broadcast_to(e['fields'].ravel(),(chains,ni*no)).copy()
        g,col=map(np.asarray,gate(ni,no,a.shape[1]).kernel(t)(keys[li],jnp.asarray(fields),jnp.asarray(a)))
        saved[f'gate_{li}']=g;saved[f'column_{li}']=col;saved[f'codes_{li}']=codes
        if li<2:
            saved[f'mask_{li}']=mask
            y=np.concatenate([decode(g[ch:ch+1],prepared[ch][3],codes[ch],mask[ch],t,0.,span,e['center'],e['radius'],e['bias']) for ch in range(chains)],axis=0)
            h=pool(np.maximum(y,0))
        else:
            qx=0.+span*codes.astype(float)/255
            sampled=readout(g,col[:,None,:],t,0.,span,0)
            h=qx.sum(-1)[:,:,None]*(e['center']-e['radius'])+2*e['radius']*sampled+e['bias']
        saved[f'output_{li}']=h
    return saved
