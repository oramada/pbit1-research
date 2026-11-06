"""Four free saturated-field positional p-bits per normalized kernel weight.

Internal research comparator, not an asserted Extropic hardware weight format.
Observer composes bit-gated integer sums, never operational per-weight counts.
"""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
from results.thrml_spw.fashion_cnn_native.inference.fast_native import vector_patches
from results.thrml_spw.fashion_cnn_native.inference.native import pool
from results.thrml_spw.convolution_native.operator import decode
from results.thrml_spw.integer_activation_native.gated_layer import activation_codes,readout

class WordCounts(AbstractObserver):
    block:object
    ni:int
    no:int
    codes:object
    def init(self):return jnp.zeros((self.codes.shape[0],self.no),jnp.int32),jnp.zeros(self.no,jnp.int32)
    def __call__(self,program,free,clamped,carry,iteration):
        z=from_global_state(block_state_to_global(free+clamped,program.gibbs_spec),program.gibbs_spec,[self.block])[0].reshape(self.ni,self.no,4)
        g=jnp.zeros_like(carry[0]);c=jnp.zeros_like(carry[1]);a=self.codes.astype(jnp.int32)
        for bit in range(4):
            g+=jnp.sum(jnp.where(z[None,:,:,bit],(a[:,:,None]<<bit),jnp.int32(0)),axis=1,dtype=jnp.int32)
            c+=jnp.sum(z[:,:,bit].astype(jnp.int32)<<bit,axis=0,dtype=jnp.int32)
        return (carry[0]+g,carry[1]+c),None

class WordLayer:
    def __init__(self,ni,no,rows):
        self.ni,self.no,self.rows=ni,no,rows;self.n=ni*no*4;self.bank=NativeProbabilityBank(np.full(self.n,.5))
    @lru_cache(None)
    def kernel(self,t):
        if not isinstance(t,int) or t<1 or 255*self.ni*15*t>=2**31:raise ValueError('Unsafe signed integer counter bound')
        def one(key,fields,codes):
            obs=WordCounts(self.bank.readout,self.ni,self.no,codes);program=set_fields(self.bank,fields)
            result,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),[jnp.zeros(self.n,jnp.bool_)],[],obs.init(),obs)
            return result
        return jax.jit(jax.vmap(one))
@lru_cache(None)
def layer(ni,no,rows):return WordLayer(ni,no,rows)

def encode(model):
    enc=[]
    for prefix in ['conv1','conv2','head']:
        w=model[prefix+'_weight'].astype(float);flat=w.reshape(w.shape[0],-1);c=(flat.min(1)+flat.max(1))/2;r=(flat.max(1)-flat.min(1))/2;q=np.divide(flat-c[:,None],r[:,None],out=np.zeros_like(flat),where=r[:,None]>0)/2+.5
        code=np.rint(np.clip(q,0,1)*15).astype(np.uint8).T.copy();bits=(code[:,:,None]>>np.arange(4))&1;fields=np.where(bits,4.,-4.).astype(np.float32)
        enc.append({'center':c,'radius':r,'fields':fields,'codes':code,'bias':model[prefix+'_bias'].astype(float)})
    return enc

def forward(x,model,enc,t,keys):
    h=np.asarray(x,float);chains,batch=h.shape[:2];saved={}
    for li,e in enumerate(enc):
        assert float(model['activation_lower'][li])==0.;span=float(model['activation_span'][li]);ni,no,_=e['fields'].shape
        if li<2:
            prepared=[vector_patches(h[ch],span) for ch in range(chains)];a=np.stack([v[0] for v in prepared]);codes=np.stack([v[1] for v in prepared]);mask=np.stack([v[2] for v in prepared])
        else:h=h.reshape(chains,batch,ni);codes=activation_codes(h,0.,span);a=codes
        gate=layer(ni,no,a.shape[1]);assert len(gate.bank.free_indices)==ni*no*4 and not len(gate.bank.clamped_indices);fields=np.broadcast_to(e['fields'].ravel(),(chains,ni*no*4)).copy();g,c=map(np.asarray,gate.kernel(t)(keys[li],jnp.asarray(fields),jnp.asarray(a)));saved[f'gate_{li}']=g;saved[f'column_{li}']=c;saved[f'codes_{li}']=codes
        if li<2:
            saved[f'mask_{li}']=mask;y=np.concatenate([decode(g[ch:ch+1],prepared[ch][3],codes[ch],mask[ch],15*t,0.,span,e['center'],e['radius'],e['bias']) for ch in range(chains)]);h=pool(np.maximum(y,0))
        else:
            qx=0.+span*codes.astype(float)/255;sampled=readout(g,c[:,None,:],15*t,0.,span,0);h=qx.sum(-1)[:,:,None]*(e['center']-e['radius'])+2*e['radius']*sampled+e['bias']
        saved[f'output_{li}']=h
    return saved
