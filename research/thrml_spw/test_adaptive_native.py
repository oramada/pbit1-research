import unittest

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_states

from .adaptive_native import AdaptiveNetwork, compact_observations, key_for, replay


class AdaptiveChecks(unittest.TestCase):
    def network(self, batch=8):
        return AdaptiveNetwork([np.array([[.2, .6], [.8, .4]], dtype=np.float32),
                                np.full((2, 2), .5, dtype=np.float32)],
                               [np.zeros(2, dtype=np.float32), np.array([1., 0.], dtype=np.float32)], batch)

    def test_moments_match_explicit_official_samples(self):
        network = self.network(1)
        bank = network.bank
        key = key_for(730, 0, 0, 0, 16)
        hidden = np.array([[.375, .625]], dtype=np.float32)
        mean, m2 = network.chunk_function(16)(jax.vmap(lambda k: k)(jax.random.split(key, 1)),
                     jnp.asarray(hidden), jnp.int32(0), jnp.zeros((1, 2)), jnp.zeros((1, 2)))
        chain_key = jax.random.split(key, 1)[0]
        states = np.asarray(sample_states(chain_key, bank.program, SamplingSchedule(1, 16, 1),
                     [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)], bank.clamp_state, [bank.readout])[0])
        values = np.array([hidden[0]@s.reshape(2, 2) for s in states], dtype=np.float64)
        np.testing.assert_allclose(np.asarray(mean)[0], values.mean(0), atol=1e-6)
        np.testing.assert_allclose(np.asarray(m2)[0], ((values-values.mean(0))**2).sum(0), atol=1e-6)

    def test_online_compaction_matches_full_path_replay_and_counts_padding(self):
        network = self.network()
        x = np.concatenate([np.zeros((9, 2)), np.ones((8, 2))]).astype(np.float32)
        hidden, first_observations = network.hidden(x, 9730, 0, 0)
        history, moments, full = network.collect(hidden, 9730, 0, 0)
        expected, expected_t = replay(history, moments, 8.)
        online, stop_t, counters = network.online(hidden, 9730, 0, 0, 8.)
        np.testing.assert_array_equal(stop_t, expected_t)
        np.testing.assert_array_equal(online, expected)
        np.testing.assert_array_equal(stop_t[:9], np.full(9, 16))
        self.assertTrue(np.any(stop_t[9:] > 16))
        executed = sum(c["chain_observations"] for c in counters)
        self.assertEqual(executed, compact_observations(stop_t, 8))
        self.assertLess(executed, sum(c["chain_observations"] for c in full))
        self.assertGreaterEqual(executed, int(stop_t.sum()))
        self.assertEqual(first_observations, 24*64)


if __name__ == "__main__":
    unittest.main()
