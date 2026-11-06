"""Direct w=p encoding using official THRML factors and Gibbs updates.

THRML SpinNode state False/True corresponds to spin -1/+1. For an isolated
spin with energy -beta*b*s, P(True)=sigmoid(2*beta*b). Thus
b=logit(w)/(2*beta). No NumPy/JAX Bernoulli replacement sampler is used here.

Endpoints use THRML clamped nodes; finite-temperature finite-bias free spins
cannot have exactly zero or one probability. This policy is explicit metadata.
"""

from __future__ import annotations

from functools import lru_cache
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from thrml import (Block, BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule,
                   SpinNode, sample_states)
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional


class NativeProbabilityBank:
    """Exactly one logical SpinNode per scalar in [0,1].

    The simulator's floating-point bias fields remain stored control parameters;
    this class does not claim the Python object compresses float storage.
    """

    def __init__(self, weights, beta: float = 1.0):
        p = np.asarray(weights, dtype=np.float64)
        if p.size == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("weights must be a nonempty finite array in [0,1]")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        self.shape = p.shape
        self.probabilities = p.reshape(-1).copy()
        self.beta = float(beta)
        self.free_indices = np.flatnonzero((self.probabilities > 0) & (self.probabilities < 1))
        self.clamped_indices = np.flatnonzero((self.probabilities == 0) | (self.probabilities == 1))
        pf = self.probabilities[self.free_indices]
        self.biases = ((np.log(pf) - np.log1p(-pf)) / (2 * beta)).astype(np.float32)
        self.nodes = [SpinNode() for _ in self.probabilities]
        self.readout = Block(self.nodes)
        self.free_blocks = ([Block([self.nodes[i] for i in self.free_indices])]
                            if self.free_indices.size else [])
        self.clamped_blocks = ([Block([self.nodes[i] for i in self.clamped_indices])]
                               if self.clamped_indices.size else [])
        self.clamp_state = ([jnp.asarray(self.probabilities[self.clamped_indices], dtype=jnp.bool_)]
                            if self.clamped_indices.size else [])
        spec = BlockGibbsSpec(self.free_blocks, self.clamped_blocks,
                             {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)})
        # IsingEBM.factors in the pinned version constructs an invalid empty edge
        # factor when edges=[]. Public unary factors avoid that upstream issue
        # without changing THRML's conditional sampler or introducing dummy nodes.
        factors = ([SpinEBMFactor(self.free_blocks, jnp.asarray(beta * self.biases))]
                   if self.free_indices.size else [])
        self.program = FactorSamplingProgram(
            spec, [SpinGibbsConditional() for _ in self.free_blocks], factors, [])

    @property
    def logical_pbits(self) -> int:
        return len(self.nodes)

    @property
    def realized_probabilities(self):
        """Analytical target after rounding control fields to float32."""
        p = self.probabilities.copy()
        field = 2 * self.beta * self.biases.astype(np.float64)
        p[self.free_indices] = np.exp(-np.logaddexp(0, -field))
        return p.reshape(self.shape)

    @lru_cache(maxsize=8)
    def _sampler(self, n_samples: int, n_chains: int, initial_one: bool):
        schedule = SamplingSchedule(n_warmup=1, n_samples=n_samples, steps_per_sample=1)
        initial = ([jnp.full((self.free_indices.size,), initial_one, dtype=jnp.bool_)]
                   if self.free_indices.size else [])

        def one_chain(key):
            return sample_states(key, self.program, schedule, initial,
                                 self.clamp_state, [self.readout])[0]

        # THRML updates one chain at a time. vmap adds independent chains, using
        # independent split keys; no unsupported leading state dimensions.
        return jax.jit(jax.vmap(one_chain))

    def sample(self, seed: int, n_samples: int, n_chains: int = 1,
               initial_one: bool = False) -> np.ndarray:
        if n_samples < 1 or n_chains < 1:
            raise ValueError("n_samples and n_chains must be positive")
        keys = jax.random.split(jax.random.key(seed), n_chains)
        result = self._sampler(n_samples, n_chains, initial_one)(keys)
        return np.asarray(result).reshape((n_chains, n_samples, *self.shape))

    def controls(self) -> dict:
        return {"logical_pbits": self.logical_pbits,
                "free_pbits": int(self.free_indices.size),
                "clamped_endpoint_pbits": int(self.clamped_indices.size),
                "bias_dtype": "float32", "bias_storage_bytes_simulator": int(self.biases.nbytes),
                "beta": self.beta, "couplings": 0,
                "backend": "thrml.FactorSamplingProgram / SpinGibbsConditional / JaxBackend",
                "endpoint_policy": "official THRML clamped nodes; no finite-bias endpoint claim"}
