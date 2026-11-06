"""Use the established native direct-sum kernel with explicit range group size."""
import numpy as np
import jax.numpy as jnp
from .calibrated_transformer_native import CalibratedGroupedLinear
from .native import NativeProbabilityBank
from .localized_controls import decode,expand

class LocalizedLinear(CalibratedGroupedLinear):
    def __init__(self,path,group):
        with np.load(path) as a:
            if 'group_size' in a.files:assert int(a['group_size'])==group
            self.codes=a['codes'];self.shape=self.codes.shape;self.lower,self.span=expand(a,group);self.group_lower=a['lower'];self.group_span=a['span'];self.bias=jnp.asarray(a['bias']);self.nominal=decode(self.codes).reshape(-1)
        self.bank=NativeProbabilityBank(np.full(self.shape,.5));assert self.bank.logical_pbits==len(self.bank.free_indices) and not len(self.bank.clamped_indices)
