import unittest

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_states

from .adaptive_native import replay
from .adaptive_shared import SharedAdaptiveNetwork


class SharedAdaptiveChecks(unittest.TestCase):
    def network(self):
        return SharedAdaptiveNetwork([np.full((2, 2), .5, dtype=np.float32)]*2,
                                     [np.zeros(2, dtype=np.float32), np.array([1., 0.], dtype=np.float32)], 8)

    def test_shared_moments_match_explicit_bank_trajectory(self):
        net = self.network()
        key, hidden = jax.random.key(731), np.array([[.125, .375], [.625, .875]], dtype=np.float32)
        mean, m2 = net.chunk_function(16)(key, jnp.asarray(hidden), jnp.int32(0), jnp.zeros((2, 2)), jnp.zeros((2, 2)))
        bank = net.bank
        states = np.asarray(sample_states(key, bank.program, SamplingSchedule(1, 16, 1),
            [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)], bank.clamp_state, [bank.readout])[0])
        values = np.array([hidden@s.reshape(2, 2) for s in states], dtype=np.float64)
        np.testing.assert_allclose(mean, values.mean(0), atol=1e-6)
        np.testing.assert_allclose(m2, ((values-values.mean(0))**2).sum(0), atol=1e-6)

    def test_no_stop_equals_fixed_and_early_stop_skips_bank_calls(self):
        net = self.network()
        hidden = np.zeros((17, 2), dtype=np.float32)
        fixed, _, counts = net.collect(hidden, 739, 0, 0)
        online, stop, actual, history, moments = net.online(hidden, 739, 0, 0, np.inf)
        np.testing.assert_array_equal(online, fixed[-1])
        self.assertEqual(sum(c["bank_observations"] for c in actual), 3*256)
        early, early_t, early_counts, traces, trace_m2 = net.online(hidden, 739, 0, 0, 3.)
        np.testing.assert_array_equal(early_t, np.full(17, 16))
        self.assertEqual(len(early_counts), 1)
        self.assertEqual(early_counts[0]["bank_observations"], 3*16)
        self.assertEqual(early_counts[0]["padded_cases"], 24)
        expected, expected_t = replay(traces, trace_m2, 3.)
        np.testing.assert_array_equal(early, expected)
        np.testing.assert_array_equal(early_t, expected_t)


if __name__ == "__main__":
    unittest.main()
