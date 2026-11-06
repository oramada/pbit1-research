"""Explicit spatial replicas, not positional bits or an Extropic weight baseline."""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from thrml import AbstractObserver,FactorSamplingProgram,SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .native import NativeProbabilityBank
from .packed_controls import load_controls


class ReplicaSum(AbstractObserver):
    readout:object
    inputs:jax.Array
    span:jax.Array
    shape:tuple
    def init(self):return jnp.zeros((self.shape[0],self.inputs.shape[0],self.shape[-1]),dtype=jnp.float32)
    def __call__(self,program,state_free,state_clamped,carry,iteration):
        state=block_state_to_global(state_free+state_clamped,program.gibbs_spec)
        bits=from_global_state(state,program.gibbs_spec,[self.readout])[0].reshape(self.shape).astype(jnp.float32)
        partial=jnp.einsum('ni,rio->rno',self.inputs,bits*self.span[None],precision=jax.lax.Precision.HIGHEST)
        return carry+partial,None


class ReplicatedNativeLinear:
    def __init__(self,artifact,replicas):
        if not isinstance(replicas,int) or replicas<1:raise ValueError('Positive replica count required')
        data=load_controls(artifact);shape=tuple(data['header']['shape']);block=data['header']['block_size']
        self.replicas=replicas;self.unique_shape=shape
        self.lower=np.repeat(data['lower'],block,axis=0)[:shape[0]];self.span=np.repeat(data['span'],block,axis=0)[:shape[0]]
        self.group_count=data['lower'].size;self.bias=jnp.asarray(data['bias'])
        probabilities=np.broadcast_to(data['probabilities'][None],(replicas,*shape)).copy()
        self.bank=NativeProbabilityBank(probabilities);self.bank.biases=np.tile(data['fields'],replicas)
        factors=[SpinEBMFactor(self.bank.free_blocks,jnp.asarray(self.bank.biases))] if len(self.bank.free_indices) else []
        self.bank.program=FactorSamplingProgram(self.bank.program.gibbs_spec,[SpinGibbsConditional() for _ in self.bank.free_blocks],factors,[])
    def controls(self):
        info=self.bank.controls();info.update(replicas=self.replicas,unique_encoded_weights=int(np.prod(self.unique_shape)),
            unique_range_metadata_bits=64*self.group_count,
            mechanism='Distinct independent SpinNodes, identical probability per weight across replicas; not positional encoding',
            sharing='Software fields are replicated. A physical shared bias fanout is only a possibility, not an implemented circuit.')
        return info
    @lru_cache(maxsize=8)
    def compiled(self,t):
        if t<1:raise ValueError('Positive T required')
        bank=self.bank;initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)] if len(bank.free_indices) else []
        lower,span=jnp.asarray(self.lower),jnp.asarray(self.span)
        def run(key,x):
            observer=ReplicaSum(bank.readout,x,span,bank.shape)
            total,_=sample_with_observation(key,bank.program,SamplingSchedule(1,t,1),initial,bank.clamp_state,observer.init(),observer)
            return total.sum(0)/(t*self.replicas)+x@lower+self.bias
        return jax.jit(run)
