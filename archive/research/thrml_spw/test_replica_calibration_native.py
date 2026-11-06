import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from .replica_calibration_native import ReplicaCalibrationLinear
from .calibration_count_native import CalibrationNativeLinear
from .packed_controls import save_controls


class ReplicaCalibrationChecks(unittest.TestCase):
    def test_native_counts_and_inference_use_every_distinct_replica_in_correct_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'controls.npz'
            w = np.array([[-.3,.4,.2],[.1,.6,.4],[.6,-.2,.5],[.4,.8,.3],[.2,.1,.6],[.5,.7,.8]], np.float32)
            b = np.array([.1, -.2, .3], np.float32); save_controls(path, w, b, 4)
            x = jnp.array([[.2,-.5,.7,.3,-.1,.4]])
            for k in [1, 4]:
                layer = ReplicaCalibrationLinear(path, k); bank = layer.bank
                fields = np.linspace(-.8, .9, len(bank.free_indices), dtype=np.float32)
                public = FactorSamplingProgram(bank.program.gibbs_spec, [SpinGibbsConditional()],
                    [SpinEBMFactor(bank.free_blocks, jnp.asarray(fields))], [])
                for t in [1, 17, 64]:
                    key = jax.random.key(1200+t)
                    bits = np.asarray(sample_states(key, public, SamplingSchedule(1, t, 1),
                        [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)], bank.clamp_state,
                        [bank.readout])[0]).reshape(t, -1)
                    counts = np.asarray(layer.compiled_counts(t)(key, jnp.asarray(fields)))
                    np.testing.assert_array_equal(counts, bits[:, bank.free_indices].sum(0))
                    weight = layer.lower+layer.span*bits.reshape(t, k, *w.shape).mean((0,1))
                    expected = np.asarray(x)@weight+b
                    np.testing.assert_allclose(layer.compiled_with_fields(t)(key, x, jnp.asarray(fields)), expected, atol=3e-6)
                    self.assertEqual(counts.dtype, np.int32)
                    self.assertEqual(bank.logical_pbits, k*w.size)
                    if k == 1:
                        old = CalibrationNativeLinear(path, 1)
                        np.testing.assert_array_equal(counts, old.compiled_counts(t)(key, jnp.asarray(fields)))


if __name__ == '__main__': unittest.main()
