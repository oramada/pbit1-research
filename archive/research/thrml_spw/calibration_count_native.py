"""Native per-free-device calibration counts, no replacement Bernoulli sampler."""
from functools import lru_cache
import equinox as eqx
import jax
import jax.numpy as jnp
from thrml import AbstractObserver, SamplingSchedule, sample_with_observation
from .controlled_replica import ControlledReplicaLinear


class FreeCounts(AbstractObserver):
    size: int

    def init(self):
        return jnp.zeros(self.size, dtype=jnp.int32)

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        return carry+state_free[0].astype(jnp.int32), None


class CalibrationNativeLinear(ControlledReplicaLinear):
    @lru_cache(maxsize=4)
    def compiled_counts(self, samples):
        if self.replicas != 1 or not isinstance(samples, int) or not 1 <= samples < 2**31:
            raise ValueError('Counts require K1 and a positive int32-safe sample count')
        bank = self.bank
        if not len(bank.free_indices):
            raise ValueError('Calibration requires free nodes')
        assert len(bank.program.per_block_interactions) == len(bank.program.per_block_interactions[0]) == 1
        initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)]

        def run(key, fields):
            if fields.shape != bank.biases.shape or fields.dtype != jnp.float32:
                raise ValueError('One float32 field per free node required')
            program = eqx.tree_at(lambda p: p.per_block_interactions[0][0].weights,
                bank.program, fields[:, None])
            observer = FreeCounts(len(bank.free_indices))
            counts, _ = sample_with_observation(key, program,
                SamplingSchedule(1, samples, 1), initial, bank.clamp_state,
                observer.init(), observer)
            return counts

        return jax.jit(run)
