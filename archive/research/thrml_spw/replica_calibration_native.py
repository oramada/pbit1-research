"""Official THRML calibration counts for explicit K1 or spatial replica banks."""
from functools import lru_cache
import equinox as eqx
import jax
import jax.numpy as jnp
from thrml import SamplingSchedule, sample_with_observation
from .controlled_replica import ControlledReplicaLinear
from .calibration_count_native import FreeCounts


class ReplicaCalibrationLinear(ControlledReplicaLinear):
    @lru_cache(maxsize=8)
    def compiled_counts(self, samples):
        if not isinstance(samples, int) or not 1 <= samples < 2**31:
            raise ValueError('Positive int32-safe sample count required')
        bank = self.bank
        if not len(bank.free_indices):
            raise ValueError('Calibration requires free nodes')
        assert len(bank.program.per_block_interactions) == len(bank.program.per_block_interactions[0]) == 1
        assert bank.program.per_block_interactions[0][0].weights.shape == (len(bank.free_indices), 1)
        initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)]

        def run(key, fields):
            if fields.shape != bank.biases.shape or fields.dtype != jnp.float32:
                raise ValueError('One float32 field per free replica node required')
            program = eqx.tree_at(lambda p: p.per_block_interactions[0][0].weights,
                bank.program, fields[:, None])
            observer = FreeCounts(len(bank.free_indices))
            counts, _ = sample_with_observation(key, program,
                SamplingSchedule(1, samples, 1), initial, bank.clamp_state,
                observer.init(), observer)
            return counts

        return jax.jit(run)
