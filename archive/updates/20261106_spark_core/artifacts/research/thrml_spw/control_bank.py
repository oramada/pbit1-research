"""Finite-resolution bias programming for the official THRML probability bank.

The bias range here is dimensionless Ising b, not a voltage or a measured device
specification. A B-bit code selects one of 2**B equally spaced finite fields.
Endpoint clamping remains explicit and separate from finite bias programming.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from thrml import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional

from .native import NativeProbabilityBank


class ControlledProbabilityBank(NativeProbabilityBank):
    def __init__(self, weights, bits: int, bound: float, beta: float=1.0):
        if not isinstance(bits,int) or not 2 <= bits <= 24:
            raise ValueError("bits must be an integer in [2,24]")
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be positive and finite")
        super().__init__(weights,beta)
        self.bits=bits
        self.bound=float(bound)
        requested_bias=self.biases.astype(np.float64)
        self.clipped_fields=int(np.count_nonzero(np.abs(requested_bias)>bound))
        levels=(1<<bits)-1
        self.codes=np.rint((np.clip(requested_bias,-bound,bound)+bound)*levels/(2*bound)).astype(np.uint32)
        self.biases=(-bound+self.codes.astype(np.float64)*(2*bound/levels)).astype(np.float32)
        # Construct a fresh bank before any JIT cache exists, and rebuild its
        # official factors from the actual quantized controls sent to THRML.
        factors=([SpinEBMFactor(self.free_blocks,jnp.asarray(beta*self.biases))]
                 if self.free_indices.size else [])
        self.program=FactorSamplingProgram(self.program.gibbs_spec,
            [SpinGibbsConditional() for _ in self.free_blocks],factors,[])

    def controls(self):
        result=super().controls()
        result.update({"field_bits":self.bits,"field_bound":self.bound,
            "field_levels":1<<self.bits,"field_spacing":2*self.bound/((1<<self.bits)-1),
            "range_clipped_fields":self.clipped_fields,
            "hypothetical_packed_controller_bits":self.bits*len(self.free_indices),
            "hypothetical_endpoint_mode_bits":2*self.logical_pbits,
            "hypothetical_shared_range_and_beta_bits":64,
            "control_model":"uniform dimensionless Ising-b DAC; not a measured physical voltage transfer"})
        return result
