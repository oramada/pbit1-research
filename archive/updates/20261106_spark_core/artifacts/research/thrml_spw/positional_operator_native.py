"""Direct sample-and-accumulate positional operators using official THRML updates."""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,SamplingSchedule,sample_with_observation,block_state_to_global,from_global_state
from .native import NativeProbabilityBank
from .controlled_positive_native import set_fields

class PositionalSum(AbstractObserver):
    readout:object
    inputs:jax.Array
    span:jax.Array
    shape:tuple
    bits:int
    def init(self):return jnp.zeros((self.inputs.shape[0],self.shape[1]),jnp.float32)
    def __call__(self,program,free,clamped,carry,iteration):
        state=block_state_to_global(free+clamped,program.gibbs_spec);z=from_global_state(state,program.gibbs_spec,[self.readout])[0].reshape(self.shape+(self.bits,)).astype(jnp.float32)
        value=jnp.sum(z*jnp.asarray(2.**np.arange(self.bits),jnp.float32),axis=-1)/((1<<self.bits)-1)
        return carry+self.inputs@(value*self.span),None

class PositionalOperator:
    def __init__(self,shape,k):
        self.shape=tuple(shape);self.bits=k;self.bank=NativeProbabilityBank(np.full(self.shape+(k,),.5));self.initial=[jnp.zeros(self.bank.logical_pbits,jnp.bool_)]
        assert len(self.bank.free_indices)==self.bank.logical_pbits and not len(self.bank.clamped_indices)
    @lru_cache(maxsize=8)
    def forward(self,t):
        def run(key,x,fields,lower,span,bias):
            program=set_fields(self.bank,fields);obs=PositionalSum(self.bank.readout,x,span,self.shape,self.bits);total,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),self.initial,[],obs.init(),obs)
            return total/t+x@lower+bias
        return jax.jit(run)
