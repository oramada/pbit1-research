"""Exact integer observable factorization for software simulation only.

For fixed integer X, sum_t X Z_t = X sum_t Z_t. This changes the observer's
simulator workspace to one int32 count per sampled weight, then performs one
integer contraction. It does not reduce stochastic samples, prove a better
physical datapath, or remove the counter storage cost of such a physical design.
"""
import jax
import jax.numpy as jnp
from thrml import (AbstractObserver, SamplingSchedule, sample_with_observation,
                   sample_states, block_state_to_global, from_global_state)
from research.thrml_spw.controlled_positive_native import set_fields


class BankCounts(AbstractObserver):
    block: object
    size: int

    def init(self):
        return jnp.zeros(self.size, dtype=jnp.int32)

    def __call__(self, program, free, clamped, carry, iteration):
        state = block_state_to_global(free + clamped, program.gibbs_spec)
        z = from_global_state(state, program.gibbs_spec, [self.block])[0]
        return carry + z.astype(jnp.int32), None


def build_count_functions(bank, samples, depthwise):
    size = min(128, bank.shape[0]); groups = bank.shape[0] // size
    assert bank.shape[1] == 64 and bank.shape[0] % size == 0
    assert samples > 0 and 2047 * size * samples < 2 ** 31
    def forward(key, inputs, fields):
        observer = BankCounts(bank.readout, bank.logical_pbits)
        initial = [jnp.zeros(bank.logical_pbits, dtype=jnp.bool_)]
        counts, _ = sample_with_observation(key, set_fields(bank, fields),
            SamplingSchedule(1, samples, 1), initial, [], observer.init(), observer)
        z = counts.reshape(groups, size, 64)
        if depthwise:
            x = inputs.reshape(inputs.shape[0], groups, size, 64)
            total = jnp.sum(x * z[None], axis=2, dtype=jnp.int32)
        else:
            total = jnp.einsum('bgk,gko->bgo', inputs.reshape(inputs.shape[0], groups, size), z)
        return total, counts
    def public(key, fields):
        initial = [jnp.zeros(bank.logical_pbits, dtype=jnp.bool_)]
        return sample_states(key, set_fields(bank, fields), SamplingSchedule(1, samples, 1),
            initial, [], [bank.readout])[0]
    return {'serial': jax.jit(forward), 'batched': jax.jit(jax.vmap(forward, in_axes=(0, 0, 0))),
            'public': jax.jit(public)}
