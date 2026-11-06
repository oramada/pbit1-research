"""Native count-only layers for the planned two-layer classifier.

External float64 readout and ReLU are separate from the temporal observers.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state
from .native import NativeProbabilityBank
from .controlled_positive_native import set_fields
class WeightCounts(AbstractObserver):
    readout:object
    n:int
    def init(self):return jnp.zeros(self.n,jnp.int32)
    def __call__(self,program,free,clamped,carry,iteration):
        z=from_global_state(block_state_to_global(free+clamped,program.gibbs_spec),program.gibbs_spec,[self.readout])[0]
        return carry+z.astype(jnp.int32),None
class BooleanCounts(AbstractObserver):
    readout:object
    ni:int
    no:int
    batch:int
    def init(self):return jnp.zeros((self.batch,self.no),jnp.int32),jnp.zeros(self.no,jnp.int32)
    def __call__(self,program,free,clamped,carry,iteration):
        z=from_global_state(block_state_to_global(free+clamped,program.gibbs_spec),program.gibbs_spec,[self.readout])[0]
        n=self.ni*self.no;w=z[:n].reshape(self.ni,self.no);a=z[n:].reshape(self.batch,self.ni)
        return (carry[0]+jnp.sum(a[:,:,None]&w[None,:,:],axis=1,dtype=jnp.int32),carry[1]+jnp.sum(w,axis=0,dtype=jnp.int32)),None
class CountLayer:
    def __init__(self,ni,no,batch=8,boolean=False):
        if min(ni,no,batch)<1:raise ValueError('positive dimensions required')
        self.ni,self.no,self.batch,self.boolean=ni,no,batch,boolean
        self.n=ni*no+(batch*ni if boolean else 0)
        self.bank=NativeProbabilityBank(np.full(self.n,.5))
    @lru_cache(None)
    def kernel(self,t):
        if not isinstance(t,int) or t<1 or t*self.ni>=2**31:raise ValueError('safe positive sample count required')
        bank=self.bank;obs=BooleanCounts(bank.readout,self.ni,self.no,self.batch) if self.boolean else WeightCounts(bank.readout,self.n)
        def run(key,fields):
            program=set_fields(bank,fields)
            result,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),[jnp.zeros(self.n,jnp.bool_)],[],obs.init(),obs)
            return result
        return jax.jit(jax.vmap(run))
def readout_weight(counts,t,x,bias):
    return np.asarray(x,float)@(np.asarray(counts,float)/t)+np.asarray(bias,float)
def readout_boolean(counts,column_counts,t,lower,span,bias):
    return float(lower)*np.asarray(column_counts,float)/t+float(span)*np.asarray(counts,float)/t+np.asarray(bias,float)
