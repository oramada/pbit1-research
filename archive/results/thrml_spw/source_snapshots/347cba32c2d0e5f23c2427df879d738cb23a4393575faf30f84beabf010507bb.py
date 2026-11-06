"""One sampled p-bit per weight, deterministic8-bit activation gated sums.

The neural temporal observer uses integer conditional gates/sums only. THRML's
sampler still simulates physical probabilities with floating control arithmetic.
"""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
class IntegerGateCounts(AbstractObserver):
    readout:object
    ni:int
    no:int
    codes:object
    def init(self):return jnp.zeros((self.codes.shape[0],self.no),jnp.int32),jnp.zeros(self.no,jnp.int32)
    def __call__(self,program,free,clamped,carry,iteration):
        z=from_global_state(block_state_to_global(free+clamped,program.gibbs_spec),program.gibbs_spec,[self.readout])[0];w=z.reshape(self.ni,self.no)
        terms=jnp.where(w[None,:,:],self.codes.astype(jnp.int32)[:,:,None],jnp.int32(0));gated=jnp.sum(terms,axis=1,dtype=jnp.int32);column=jnp.sum(w,axis=0,dtype=jnp.int32)
        return (carry[0]+gated,carry[1]+column),None
class IntegerGateLayer:
    def __init__(self,ni,no,batch=8):
        if min(ni,no,batch)<1:raise ValueError('positive dimensions required')
        self.ni,self.no,self.batch=ni,no,batch;self.n=ni*no;self.bank=NativeProbabilityBank(np.full(self.n,.5))
    @lru_cache(None)
    def kernel(self,t):
        if not isinstance(t,int) or t<1 or 255*self.ni*t>=2**31:raise ValueError('sample count exceeds signed native counter bound')
        def one(key,field,codes):
            observer=IntegerGateCounts(self.bank.readout,self.ni,self.no,codes);program=set_fields(self.bank,field)
            result,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),[jnp.zeros(self.n,jnp.bool_)],[],observer.init(),observer)
            return result
        return jax.jit(jax.vmap(one))
def activation_codes(x,lower,span):
    if not np.isfinite(span) or span<=0 or not np.isfinite(lower):raise ValueError('finite nonconstant activation range required')
    a=(np.asarray(x,float)-float(lower))/float(span)
    if not np.isfinite(a).all():raise ValueError('finite inputs required')
    return np.rint(255*np.clip(a,0,1)).astype(np.uint8)
def readout(gated,column,t,lower,span,bias):
    return float(lower)*np.asarray(column,float)/t+(float(span)/255)*np.asarray(gated,float)/t+np.asarray(bias,float)
