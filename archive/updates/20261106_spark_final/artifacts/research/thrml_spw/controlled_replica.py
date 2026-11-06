"""Dynamic unary fields on a fixed, explicit official THRML replica graph.

Only the unary interaction's weight array is replaced. SpinGibbsConditional,
the graph, Gibbs schedule, endpoint clamps and observer remain official/native.
This layout is deliberately tied to the pinned unary-only THRML program and
checked against separately constructed public factors in the trajectory test.
"""
from functools import lru_cache
import equinox as eqx
import jax
import jax.numpy as jnp
from thrml import SamplingSchedule, sample_with_observation
from .replicated_native import ReplicatedNativeLinear, ReplicaSum


class ControlledReplicaLinear(ReplicatedNativeLinear):
    @lru_cache(maxsize=8)
    def compiled_with_fields(self, samples):
        if not isinstance(samples, int) or samples < 1:
            raise ValueError('Positive integer sample count required')
        bank = self.bank
        if not len(bank.free_indices):
            raise ValueError('This stress-test operator requires free nodes')
        interactions = bank.program.per_block_interactions
        assert len(interactions) == len(interactions[0]) == 1
        assert interactions[0][0].weights.shape == (len(bank.free_indices), 1)
        initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)]
        lower, span = jnp.asarray(self.lower), jnp.asarray(self.span)

        def run(key, x, fields):
            if fields.shape != bank.biases.shape or fields.dtype != jnp.float32:
                raise ValueError('One float32 field per free replica node required')
            program = eqx.tree_at(
                lambda p: p.per_block_interactions[0][0].weights,
                bank.program, fields[:, None])
            observer = ReplicaSum(bank.readout, x, span, bank.shape)
            total, _ = sample_with_observation(
                key, program, SamplingSchedule(1, samples, 1), initial,
                bank.clamp_state, observer.init(), observer)
            return total.sum(0)/(samples*self.replicas)+x@lower+self.bias

        return jax.jit(run)
