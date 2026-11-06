"""Native affine weight banks with per-output, local input-block ranges.

Each weight still has one logical p-bit. Shared float32 lower/span values and
external multiplication are explicit extra resources, not free precision.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from thrml import AbstractObserver, SamplingSchedule, block_state_to_global, from_global_state, sample_with_observation
from .native import NativeProbabilityBank


class GroupedSum(AbstractObserver):
    readout: object
    inputs: jax.Array
    span: jax.Array
    weight_shape: tuple

    def init(self):
        return jnp.zeros((self.inputs.shape[0], self.weight_shape[1]), dtype=jnp.float32)

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        state = block_state_to_global(state_free+state_clamped, program.gibbs_spec)
        z = from_global_state(state, program.gibbs_spec, [self.readout])[0]
        return carry+self.inputs@(z.reshape(self.weight_shape).astype(jnp.float32)*self.span), None


class GroupedRangeNativeLinear:
    def __init__(self, weight, bias, block_size=128):
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2 or not w.size or not np.isfinite(w).all():
            raise ValueError('Nonempty finite matrix required')
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError('Positive integer block size required')
        self.block_size = block_size
        self.group_lower = np.stack([w[i:i+block_size].min(0) for i in range(0,len(w),block_size)])
        self.group_span = np.stack([w[i:i+block_size].max(0) for i in range(0,len(w),block_size)])-self.group_lower
        self.lower = np.repeat(self.group_lower,block_size,axis=0)[:len(w)]
        self.span = np.repeat(self.group_span,block_size,axis=0)[:len(w)]
        p = np.zeros_like(w)
        np.divide(w-self.lower,self.span,out=p,where=self.span!=0)
        self.bank = NativeProbabilityBank(np.clip(p,0,1))
        self.bias = jnp.asarray(bias,dtype=jnp.float32)
        if self.bias.shape != (w.shape[1],):raise ValueError('Bias shape mismatch')

    @property
    def realized_weights(self):
        return self.lower+self.span*self.bank.realized_probabilities

    def controls(self):
        info = self.bank.controls()
        info.update({'mapping':'input-block per-output lower+span*p; not direct p=w',
                     'input_block_size':self.block_size,'groups':self.group_lower.size,
                     'metadata_float32_values':2*self.group_lower.size,'metadata_bits':64*self.group_lower.size,
                     'constant_groups':int(np.sum(self.group_span==0)),
                     'expanded_range_buffer_bytes_simulator':self.lower.nbytes+self.span.nbytes,
                     'deterministic_bias_values':int(self.bias.size),
                     'arithmetic':'External float32 sample-by-span multiplication and dense linear accumulation; deterministic input@lower correction once per invocation'})
        return info

    @lru_cache(maxsize=8)
    def compiled(self,t):
        bank=self.bank
        if t<1:raise ValueError('Positive sample count required')
        initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)] if len(bank.free_indices) else []
        lower,span=jnp.asarray(self.lower),jnp.asarray(self.span)
        def run(key,inputs):
            observer=GroupedSum(bank.readout,inputs,span,bank.shape)
            total,_=sample_with_observation(key,bank.program,SamplingSchedule(1,t,1),initial,bank.clamp_state,observer.init(),observer)
            return total/t+inputs@lower+self.bias
        return jax.jit(run)
