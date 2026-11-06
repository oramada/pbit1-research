"""Independent public-state full-cohort replay with indexed spatial patches."""
import numpy as np
import jax.numpy as jnp
from results.thrml_spw.fashion_cnn_native.compact.recover import public_draw,independent_parameters,pooling
def patches(x,span):
    n,c,h,w=x.shape;quant=np.rint(np.clip(x/span,0,1)*255).astype(np.uint8);codes=np.empty((n,h,w,c*9),np.uint8);mask=np.empty_like(codes);rr,cc=np.indices((h,w))
    for ci in range(c):
        for kr in range(3):
            for kc in range(3):
                ir,ic=rr+kr-1,cc+kc-1;valid=(ir>=0)&(ir<h)&(ic>=0)&(ic<w);slot=ci*9+kr*3+kc;codes[:,:,:,slot]=np.where(valid[None],quant[:,ci,np.clip(ir,0,h-1),np.clip(ic,0,w-1)],0);mask[:,:,:,slot]=valid
    return codes.reshape(n*h*w,c*9),mask.reshape(n*h*w,c*9)
def recover(x,model,t,keys):
    h=np.asarray(x,float);chains,batch=h.shape[:2];saved={}
    for li,prefix in enumerate(['conv1','conv2','head']):
        assert float(model['activation_lower'][li])==0.;span=float(model['activation_span'][li]);c,r,fields=independent_parameters(model,prefix);ni,no=fields.shape;states=np.asarray(public_draw(ni*no,t)(keys[li],jnp.asarray(np.broadcast_to(fields.ravel(),(chains,ni*no)).copy())));wc=states.sum(1,dtype=np.uint32).reshape(chains,ni,no);col=wc.sum(1,dtype=np.uint32).astype(np.int32);saved[f'column_{li}']=col
        if li<2:
            height,width=h.shape[-2:];parts=[patches(h[ch],span) for ch in range(chains)];code=np.stack([v[0] for v in parts]);mask=np.stack([v[1] for v in parts]);g=np.einsum('cpi,cio->cpo',code.astype(np.uint32),wc,dtype=np.uint32,optimize=False);m=np.einsum('cpi,cio->cpo',mask.astype(np.uint32),wc,dtype=np.uint32,optimize=False);saved[f'gate_{li}']=np.concatenate([g,m],axis=1).astype(np.int32);saved[f'codes_{li}']=code;saved[f'mask_{li}']=mask;ys=[]
            for ch in range(chains):
                sx=np.sum(mask[ch]*(0.+span*code[ch].astype(float)/255),axis=1);sample=0.*m[ch].astype(float)/t+(span/255)*g[ch].astype(float)/t;y=sx[:,None]*(c-r)+2*r*sample+model[prefix+'_bias'].astype(float);ys.append(y.reshape(batch,height,width,no).transpose(0,3,1,2))
            h=pooling(np.stack(ys))
        else:
            h=h.reshape(chains,batch,ni);a=np.rint(255*np.clip(h/span,0,1)).astype(np.uint8);g=np.einsum('cbi,cio->cbo',a.astype(np.uint32),wc,dtype=np.uint32,optimize=False).astype(np.int32);saved[f'gate_{li}']=g;saved[f'codes_{li}']=a;qx=0.+span*a.astype(float)/255;sample=0.*col[:,None,:].astype(float)/t+(span/255)*g.astype(float)/t;h=qx.sum(-1)[:,:,None]*(c-r)+2*r*sample+model[prefix+'_bias'].astype(float)
        saved[f'output_{li}']=h
    return saved
