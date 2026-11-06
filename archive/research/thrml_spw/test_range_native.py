import unittest

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_states

from .range_native import RangeNativeLinear


class RangeChecks(unittest.TestCase):
    def test_native_output_matches_same_path_affine_weight_matrix(self):
        weight = np.array([[-.3, .2, .5], [.1, .4, .5], [.6, .3, .5]], dtype=np.float32)
        bias = np.array([.1, -.2, .25], dtype=np.float32)
        inputs = np.array([[.2, -.5, .7], [.1, .4, -.2]], dtype=np.float32)
        layer = RangeNativeLinear(weight, bias)
        np.testing.assert_allclose(layer.realized_weights, weight, atol=4e-8)
        self.assertEqual(layer.controls()["constant_groups"], 1)
        self.assertEqual(layer.controls()["metadata_bits"], 192)
        self.assertTrue(np.all(layer.bank.probabilities.reshape(weight.shape)[:, 2] == 0))
        key, t = jax.random.key(2907), 64
        result = np.asarray(layer.compiled(t)(key, jnp.asarray(inputs)))
        bank = layer.bank
        states = np.asarray(sample_states(key, bank.program, SamplingSchedule(1, t, 1),
            [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)], bank.clamp_state, [bank.readout])[0])
        sampled = layer.lower+layer.span*states.reshape(t, *weight.shape)
        reference = inputs@sampled.mean(0)+bias
        np.testing.assert_allclose(result, reference, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
