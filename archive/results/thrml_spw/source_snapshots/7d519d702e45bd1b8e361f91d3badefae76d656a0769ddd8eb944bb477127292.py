"""Bounded fixed-point output accumulators driven by official THRML samples.

The sampled dot product is still calculated in float32. This model quantizes its
increment and saturates the running integer sum; it does not model a multiplier,
ADC, or physical circuit. Neural biases are added after division by T.
"""
from __future__ import annotations

from functools import lru_cache
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from thrml import AbstractObserver, SamplingSchedule, block_state_to_global, from_global_state, sample_with_observation

from .native import NativeProbabilityBank


class FixedPointSum(AbstractObserver):
    readout: object
    inputs: jax.Array
    weight_shape: tuple
    bits: int
    fractional_bits: int

    def init(self):
        return (jnp.zeros((self.inputs.shape[0], self.weight_shape[1]), dtype=jnp.int32),
                jnp.int32(0), jnp.int32(0))

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        global_state = block_state_to_global(state_free+state_clamped, program.gibbs_spec)
        z = from_global_state(global_state, program.gibbs_spec, [self.readout])[0]
        term = self.inputs@z.reshape(self.weight_shape).astype(jnp.float32)
        lo, hi = -(1 << (self.bits-1)), (1 << (self.bits-1))-1
        rounded = jnp.rint(term*(1 << self.fractional_bits))
        clipped = jnp.clip(rounded, lo, hi).astype(jnp.int32)
        total = carry[0]+clipped
        return (jnp.clip(total, lo, hi),
                carry[1]+jnp.sum((rounded < lo) | (rounded > hi), dtype=jnp.int32),
                carry[2]+jnp.sum((total < lo) | (total > hi), dtype=jnp.int32)), None


class FixedPointLinear:
    def __init__(self, weight, bias, bits, fractional_bits):
        if not 4 <= bits <= 24 or not 0 <= fractional_bits < bits:
            raise ValueError("Require 4 <= bits <= 24 and 0 <= fractional_bits < bits")
        # At most two 24-bit signed integers are added before saturation. int32
        # therefore cannot overflow before the explicit saturation operation.
        self.bank = NativeProbabilityBank(weight)
        self.bias = jnp.asarray(bias)
        self.bits, self.fractional_bits = int(bits), int(fractional_bits)

    @lru_cache(maxsize=32)
    def compiled(self, t):
        bank = self.bank
        schedule = SamplingSchedule(1, t, 1)
        initial = [jnp.zeros((len(bank.free_indices),), dtype=jnp.bool_)] if len(bank.free_indices) else []

        def run(key, inputs):
            observer = FixedPointSum(bank.readout, inputs, bank.shape, self.bits, self.fractional_bits)
            carry, _ = sample_with_observation(key, bank.program, schedule, initial,
                                               bank.clamp_state, observer.init(), observer)
            output = carry[0].astype(jnp.float32)/(t*(1 << self.fractional_bits))+self.bias
            return output, jnp.array([carry[1], carry[2]])
        return jax.jit(run)
