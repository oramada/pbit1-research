"""Full direct p=w network with dynamic native unary fields and measured counts.

Each network weight keeps one logical node; vmap creates independent inference
chains for dataset batches, not an assertion of extra physical device capacity.
"""
from functools import lru_cache
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_with_observation
from .positive_inference import NativeLinear,PreactivationSum
from .native import NativeProbabilityBank
from .calibration_count_native import FreeCounts


def set_fields(bank,fields):
    if fields.shape!=bank.biases.shape or fields.dtype!=jnp.float32:
        raise ValueError('One float32 field per free node required')
    if not len(bank.free_indices):return bank.program
    assert len(bank.program.per_block_interactions)==len(bank.program.per_block_interactions[0])==1
    return eqx.tree_at(lambda p:p.per_block_interactions[0][0].weights,bank.program,fields[:,None])


class ControlledDirectLinear(NativeLinear):
    @lru_cache(maxsize=8)
    def with_fields(self,samples):
        if not isinstance(samples,int) or samples<1:raise ValueError('Positive integer samples required')
        bank=self.bank
        initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)] if len(bank.free_indices) else []
        def run(key,x,fields):
            program=set_fields(bank,fields);observer=PreactivationSum(bank.readout,x,bank.shape)
            total,_=sample_with_observation(key,program,SamplingSchedule(1,samples,1),initial,bank.clamp_state,observer.init(),observer)
            return total/samples+self.bias
        return jax.jit(run)


class ControlledPositiveNetwork:
    def __init__(self,weights,biases):
        if len(weights)!=2 or len(biases)!=2 or weights[0].shape[1]!=weights[1].shape[0]:raise ValueError('Two compatible dense layers required')
        self.layers=[ControlledDirectLinear(w,b) for w,b in zip(weights,biases)]
        self.free_split=len(self.layers[0].bank.free_indices)
        self.calibration_bank=NativeProbabilityBank(np.concatenate([np.asarray(w).reshape(-1) for w in weights]))
        assert len(self.calibration_bank.free_indices)==sum(len(l.bank.free_indices) for l in self.layers)
        np.testing.assert_array_equal(self.calibration_bank.biases,np.concatenate([l.bank.biases for l in self.layers]))

    @lru_cache(maxsize=4)
    def counts(self,samples):
        if not isinstance(samples,int) or not 1<=samples<2**31:raise ValueError('Positive int32-safe samples required')
        bank=self.calibration_bank
        if not len(bank.free_indices):raise ValueError('Counts require free nodes')
        initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)]
        def run(key,fields):
            program=set_fields(bank,fields);observer=FreeCounts(len(bank.free_indices))
            counts,_=sample_with_observation(key,program,SamplingSchedule(1,samples,1),initial,bank.clamp_state,observer.init(),observer)
            return counts
        return jax.jit(run)

    @lru_cache(maxsize=4)
    def dataset(self,samples):
        first,last=[l.with_fields(samples) for l in self.layers];split=self.free_split
        def run(base_key,input_batches,fields):
            if fields.shape!=self.calibration_bank.biases.shape:raise ValueError('All free network fields required')
            keys=jax.vmap(lambda i:jax.random.fold_in(base_key,i))(jnp.arange(input_batches.shape[0]))
            def one(key,x):
                k0,k1=jax.random.split(key)
                pre=first(k0,x,fields[:split]);hidden=jax.nn.relu(pre)
                logits=last(k1,hidden,fields[split:])
                return pre,logits
            return jax.vmap(one)(keys,input_batches)
        return jax.jit(run)
