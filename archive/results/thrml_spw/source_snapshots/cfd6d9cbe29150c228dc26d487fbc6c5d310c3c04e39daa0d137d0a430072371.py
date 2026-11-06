import unittest
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_states

from .accumulator import FixedPointLinear


class AccumulatorChecks(unittest.TestCase):
    def test_matches_explicit_native_trajectory_integer_rounding(self):
        weights = np.array([[.2, .7], [0., 1.], [.8, .5]], dtype=np.float32)
        inputs = np.array([[.125, -.375, .625], [.25, .125, -.875]], dtype=np.float32)
        layer = FixedPointLinear(weights, np.array([.25, -.125]), 16, 2)
        key, t = jax.random.key(973), 16
        output, counts = layer.compiled(t)(key, jnp.asarray(inputs))
        bank = layer.bank
        initial = [jnp.zeros((len(bank.free_indices),), dtype=jnp.bool_)]
        states = np.asarray(sample_states(key, bank.program, SamplingSchedule(1, t, 1),
                                        initial, bank.clamp_state, [bank.readout])[0])
        carry = np.zeros((2, 2), dtype=np.int32)
        for state in states:
            carry = np.clip(carry+np.rint((inputs@state.reshape(weights.shape))*4).astype(np.int32), -32768, 32767)
        np.testing.assert_array_equal(np.asarray(output), carry/64+np.array([.25, -.125]))
        np.testing.assert_array_equal(np.asarray(counts), [0, 0])

    def test_increment_and_running_sum_saturation_are_counted(self):
        layer = FixedPointLinear(np.ones((1, 1)), np.zeros(1), 6, 0)
        output, counts = layer.compiled(4)(jax.random.key(7), jnp.array([[100.]]))
        np.testing.assert_array_equal(np.asarray(output), [[31/4]])
        np.testing.assert_array_equal(np.asarray(counts), [4, 3])


if __name__ == "__main__":
    unittest.main()
