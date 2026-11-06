"""One native p-bit per weight with explicit per-output affine metadata.

This is separate from direct w=p encoding: p=(w-lower)/span and the expected
weight is lower+span*p. Metadata and its external arithmetic must be counted.
Constant columns use p=0 clamps and retain their value in the lower metadata.
"""
from __future__ import annotations

from functools import lru_cache
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_with_observation

from .native import NativeProbabilityBank
from .positive_inference import PreactivationSum


class RangeNativeLinear:
    def __init__(self, weight, bias):
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2 or not weight.size or not np.isfinite(weight).all():
            raise ValueError("A nonempty finite 2D weight matrix is required")
        self.lower = weight.min(axis=0)
        upper = weight.max(axis=0)
        self.span = upper-self.lower
        probabilities = np.zeros_like(weight)
        np.divide(weight-self.lower, self.span, out=probabilities, where=self.span != 0)
        probabilities = np.clip(probabilities, 0, 1)
        self.bank = NativeProbabilityBank(probabilities)
        self.bias = jnp.asarray(bias, dtype=jnp.float32)

    @property
    def realized_weights(self):
        return self.lower+self.span*self.bank.realized_probabilities

    def controls(self):
        info = self.bank.controls()
        info.update({"mapping": "per-output lower+span*p; not direct p=w",
                     "groups": len(self.lower), "metadata_float32_values": 2*len(self.lower),
                     "metadata_bits": 2*len(self.lower)*32,
                     "constant_groups": int(np.sum(self.span == 0)),
                     "deterministic_bias_values": int(self.bias.size),
                     "arithmetic": "One input sum, per-output constant correction and per-output span multiplication in external float32 arithmetic"})
        return info

    @lru_cache(maxsize=32)
    def compiled(self, t):
        bank = self.bank
        schedule = SamplingSchedule(1, t, 1)
        initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)] if len(bank.free_indices) else []
        lower, span = jnp.asarray(self.lower), jnp.asarray(self.span)

        def run(key, inputs):
            observer = PreactivationSum(bank.readout, inputs, bank.shape)
            total, _ = sample_with_observation(key, bank.program, schedule, initial,
                        bank.clamp_state, observer.init(), observer)
            return (total/t)*span+jnp.sum(inputs, axis=-1, keepdims=True)*lower+self.bias
        return jax.jit(run)
