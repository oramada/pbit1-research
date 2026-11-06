"""Expectation-aware local-range execution via sampled group partial sums.

The existing weight banks and official sampler are unchanged. Group scaling is
moved after temporal accumulation, at the cost of a larger accumulator. This is
external modeled arithmetic, not a physical-chip implementation or energy claim.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from thrml import AbstractObserver,SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation
from .controlled_grouped_native import ControlledGroupedLinear


class GroupPartialSum(AbstractObserver):
    readout:object
    inputs:jax.Array
    weight_shape:tuple
    block_size:int

    def init(self):
        n,g,_=self.inputs.shape
        return jnp.zeros((n,g,self.weight_shape[1]),dtype=jnp.float32)

    def __call__(self,program,state_free,state_clamped,carry,iteration):
        state=block_state_to_global(state_free+state_clamped,program.gibbs_spec)
        bits=from_global_state(state,program.gibbs_spec,[self.readout])[0].reshape(self.weight_shape)
        padding=self.inputs.shape[1]*self.block_size-self.weight_shape[0]
        grouped=jnp.pad(bits,((0,padding),(0,0))).reshape(self.inputs.shape[1],self.block_size,self.weight_shape[1]).astype(jnp.float32)
        partial=jnp.einsum('ngb,gbo->ngo',self.inputs,grouped,precision=jax.lax.Precision.HIGHEST)
        return carry+partial,None


class PartialGroupedLinear(ControlledGroupedLinear):
    def controls(self):
        info=super().controls()
        info.update({'execution':'Sampled group partial sums; scale after temporal averaging',
                     'arithmetic':'External binary-input partial matmul and larger group accumulators; one postaccumulation span scale plus grouped deterministic lower correction'})
        return info

    def operation_ledger(self,input_rows,samples):
        groups,outputs=self.group_span.shape;n=self.bank.logical_pbits
        return {'logical_pbits':n,'samples':samples,'input_rows':input_rows,
            'binary_product_terms':input_rows*n*samples,
            'direct_sample_span_multiplications':n*samples,
            'partial_postaccumulation_span_multiplications':input_rows*groups*outputs,
            'direct_expanded_lower_product_terms':input_rows*n,
            'grouped_lower_product_terms':input_rows*groups*outputs,
            'direct_output_accumulator_float32_values':input_rows*outputs,
            'partial_output_accumulator_float32_values':input_rows*groups*outputs,
            'scope':'Algebraic counts for one bank batch, excluding code/program/init/readout and actual kernel fusion. Binary products and stochastic draws unchanged; group accumulator is larger.'}

    @lru_cache(maxsize=8)
    def compiled(self,t):
        if t<1:raise ValueError('Positive T required')
        bank=self.bank;groups,outputs=self.group_span.shape;block=self.block_size
        lower,span=jnp.asarray(self.group_lower),jnp.asarray(self.group_span)
        initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)] if len(bank.free_indices) else []
        def run(key,x):
            padding=groups*block-bank.shape[0]
            grouped=jnp.pad(x,((0,0),(0,padding))).reshape(x.shape[0],groups,block)
            observer=GroupPartialSum(bank.readout,grouped,bank.shape,block)
            total,_=sample_with_observation(key,bank.program,SamplingSchedule(1,t,1),initial,bank.clamp_state,observer.init(),observer)
            correction=jnp.sum(jnp.sum(grouped,axis=-1)[:,:,None]*lower[None],axis=1)
            return jnp.sum((total/t)*span[None],axis=1)+correction+self.bias
        return jax.jit(run)
