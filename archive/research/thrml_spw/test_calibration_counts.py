import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from .calibration_count_native import CalibrationNativeLinear
from .packed_controls import save_controls


class CountChecks(unittest.TestCase):
    def test_counts_equal_public_factor_trajectories_and_exclude_clamps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'controls.npz'
            w = np.array([[0., .4], [.2, .4], [.7, .9], [.5, .1], [.3, .8]], np.float32)
            save_controls(path, w, np.array([0., 0.], np.float32), 3)
            layer = CalibrationNativeLinear(path, 1); bank = layer.bank
            fields = np.linspace(-.3, .5, len(bank.free_indices), dtype=np.float32)
            public = FactorSamplingProgram(bank.program.gibbs_spec, [SpinGibbsConditional()],
                [SpinEBMFactor(bank.free_blocks, jnp.asarray(fields))], [])
            for t in [1, 17, 64]:
                key = jax.random.key(900+t)
                bits = np.asarray(sample_states(key, public, SamplingSchedule(1, t, 1),
                    [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)], bank.clamp_state,
                    [bank.readout])[0]).reshape(t, -1)
                counts = np.asarray(layer.compiled_counts(t)(key, jnp.asarray(fields)))
                np.testing.assert_array_equal(counts, bits[:, bank.free_indices].sum(0))
                self.assertEqual(counts.dtype, np.int32)
            self.assertGreater(len(bank.clamped_indices), 0)


if __name__ == '__main__':
    unittest.main()
