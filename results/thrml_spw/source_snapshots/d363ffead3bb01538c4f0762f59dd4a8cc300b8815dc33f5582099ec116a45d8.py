"""Program actual official THRML from packed bias controls, without target floats."""
import jax.numpy as jnp
from thrml import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .native import NativeProbabilityBank
from .grouped_range_native import GroupedRangeNativeLinear
from .packed_controls import load_controls
import numpy as np


class PackedBank(NativeProbabilityBank):
    def __init__(self,data):
        super().__init__(data['probabilities'])
        np.testing.assert_array_equal(self.free_indices,data['free_indices'])
        self.biases=data['fields'].copy();self.codes=data['codes'].copy();self.bound=data['header']['bound']
        factors=[SpinEBMFactor(self.free_blocks,jnp.asarray(self.biases))] if len(self.free_indices) else []
        self.program=FactorSamplingProgram(self.program.gibbs_spec,[SpinGibbsConditional() for _ in self.free_blocks],factors,[])
    def controls(self):
        info=super().controls();info.update(field_bits=8,field_bound=self.bound,actual_code_bytes=self.codes.nbytes,
            programming_source='Packed finite bias codes and modes; no original target FP32 weights')
        return info


class PackedNativeLinear(GroupedRangeNativeLinear):
    def __init__(self,path):
        data=load_controls(path);shape=tuple(data['header']['shape']);self.block_size=data['header']['block_size']
        self.group_lower=data['lower'];self.group_span=data['span']
        self.lower=np.repeat(self.group_lower,self.block_size,axis=0)[:shape[0]]
        self.span=np.repeat(self.group_span,self.block_size,axis=0)[:shape[0]]
        self.bank=PackedBank(data);self.bias=jnp.asarray(data['bias'])
