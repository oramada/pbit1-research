"""Vectorize independent official THRML banks without changing their keys.

This component is not yet integrated into any full-model study. Vmap lanes are
independent executions of the same bank, not extra p-bits for one parameter.
"""
import jax
import jax.numpy as jnp
from thrml import SamplingSchedule, sample_states, sample_with_observation
from research.thrml_spw.controlled_positive_native import set_fields
from .programmed_worker import GroupSum


def build_bank_functions(bank, samples, depthwise):
    size = min(128, bank.shape[0])
    assert bank.shape[1] == 64 and bank.shape[0] % size == 0
    assert samples > 0 and 2047 * size * samples < 2 ** 31
    def forward(key, inputs, fields):
        initial = [jnp.zeros(bank.logical_pbits, dtype=jnp.bool_)]
        observer = GroupSum(bank.readout, inputs, bank.shape, depthwise, size)
        total, _ = sample_with_observation(key, set_fields(bank, fields),
            SamplingSchedule(1, samples, 1), initial, [], observer.init(), observer)
        return total
    def public(key, fields):
        initial = [jnp.zeros(bank.logical_pbits, dtype=jnp.bool_)]
        return sample_states(key, set_fields(bank, fields), SamplingSchedule(1, samples, 1),
                             initial, [], [bank.readout])[0]
    return {'serial': jax.jit(forward), 'serial_public': jax.jit(public),
            'batched': jax.jit(jax.vmap(forward, in_axes=(0, 0, 0))),
            'batched_public': jax.jit(jax.vmap(public, in_axes=(0, 0)))}
