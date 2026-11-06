"""All-free affine grouped operators from finite packed programming controls."""
from functools import lru_cache
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_with_observation
from .native import NativeProbabilityBank
from .grouped_range_native import GroupedSum
from .controlled_positive_native import set_fields
from .calibration_count_native import FreeCounts


from .calibrated_transformer_controls import export_controls,decode,device


class CalibratedGroupedLinear:
    def __init__(self,path):
        a=np.load(path);self.codes=a['codes'];self.shape=self.codes.shape
        self.lower=np.repeat(a['lower'],128,axis=0)[:self.shape[0]];self.span=np.repeat(a['span'],128,axis=0)[:self.shape[0]]
        self.group_lower=a['lower'];self.group_span=a['span'];self.bias=jnp.asarray(a['bias'])
        self.nominal=decode(self.codes).reshape(-1)
        # Only controls/ranges enter runtime, no target floating weight tensor.
        self.bank=NativeProbabilityBank(np.full(self.shape,.5))
        assert self.bank.logical_pbits==len(self.bank.free_indices) and not len(self.bank.clamped_indices)

    @lru_cache(maxsize=2)
    def counts(self,samples):
        bank=self.bank;initial=[jnp.zeros(bank.logical_pbits,dtype=jnp.bool_)]
        def run(key,fields):
            program=set_fields(bank,fields);obs=FreeCounts(bank.logical_pbits)
            count,_=sample_with_observation(key,program,SamplingSchedule(1,samples,1),initial,[],obs.init(),obs)
            return count
        return jax.jit(run)

    @lru_cache(maxsize=2)
    def forward(self,samples):
        bank=self.bank;initial=[jnp.zeros(bank.logical_pbits,dtype=jnp.bool_)];lower=jnp.asarray(self.lower);span=jnp.asarray(self.span)
        def run(key,x,fields):
            program=set_fields(bank,fields);obs=GroupedSum(bank.readout,x,span,self.shape)
            total,_=sample_with_observation(key,program,SamplingSchedule(1,samples,1),initial,[],obs.init(),obs)
            return total/samples+x@lower+self.bias
        return jax.jit(run)
