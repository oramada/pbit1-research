import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from .controlled_replica import ControlledReplicaLinear
from .packed_controls import save_controls


class ControlledReplicaChecks(unittest.TestCase):
    def test_dynamic_fields_equal_separately_built_official_factor_trajectories(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'controls.npz'
            w = np.array([[-.3, .5], [.1, .5], [.6, -.2], [.4, .8], [.2, .1]], np.float32)
            b = np.array([.1, -.2], np.float32)
            save_controls(path, w, b, 3)
            for k in [1, 4]:
                layer = ControlledReplicaLinear(path, k)
                bank = layer.bank
                x = jnp.array([[.2, -.5, .7, .3, -.1], [-.4, .1, .2, -.2, .6]])
                initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)]
                for offset in [0., .3]:
                    fields = (bank.biases+np.random.default_rng(7).normal(
                        0, offset, bank.biases.shape)).astype(np.float32)
                    public = FactorSamplingProgram(bank.program.gibbs_spec,
                        [SpinGibbsConditional()],
                        [SpinEBMFactor(bank.free_blocks, jnp.asarray(fields))], [])
                    key = jax.random.key(714)
                    bits = np.asarray(sample_states(key, public, SamplingSchedule(1, 32, 1),
                        initial, bank.clamp_state, [bank.readout])[0]).reshape(32, k, *w.shape)
                    expected = np.asarray(x)@(layer.lower+layer.span*bits.mean((0, 1)))+b
                    observed = np.asarray(layer.compiled_with_fields(32)(key, x, jnp.asarray(fields)))
                    np.testing.assert_allclose(observed, expected, atol=3e-6)
                    if offset == 0:
                        np.testing.assert_array_equal(observed, layer.compiled(32)(key, x))


if __name__ == '__main__':
    unittest.main()
