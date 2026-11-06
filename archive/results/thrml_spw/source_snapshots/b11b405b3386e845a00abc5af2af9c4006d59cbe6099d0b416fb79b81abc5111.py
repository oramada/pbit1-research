import unittest
import numpy as np
from .gain_calibration_math import fit_two_probe, corrected_codes
from .calibration_control_math import decode_command


class GainCalibrationChecks(unittest.TestCase):
    def test_identifies_distinct_known_gains_offsets_and_quantizes_inverse(self):
        lo, hi = [float(decode_command(np.array(c, np.uint8))) for c in [96, 159]]
        true_gain = np.array([.7, 1., 1.5]); true_offset = np.array([-.3, .2, .1])
        n = 1_000_000
        counts = [np.rint(n/(1+np.exp(-2*(true_gain*probe+true_offset)))).astype(np.int32) for probe in [lo, hi]]
        gain, offset, raw = fit_two_probe(*counts, n, lo, hi)
        np.testing.assert_allclose(gain, true_gain, atol=1e-5)
        np.testing.assert_allclose(offset, true_offset, atol=1e-5)
        nominal = np.array([-.6, .3, 1.])
        codes, _ = corrected_codes(nominal, gain, offset)
        recovered = true_gain*decode_command(codes)+true_offset
        np.testing.assert_allclose(recovered, nominal, atol=1.5*4/255+2e-5)

    def test_declared_guard_handles_reversed_counts_and_invalid_budget(self):
        gain, offset, raw = fit_two_probe(np.array([100]), np.array([0]), 100, -1., 1.)
        self.assertLess(raw[0], 0.)
        self.assertEqual(gain[0], .25)
        self.assertTrue(np.isfinite(offset).all())
        with self.assertRaises(ValueError):
            fit_two_probe(np.array([1.5]), np.array([2.]), 5, -1., 1.)
        with self.assertRaises(ValueError):
            fit_two_probe(np.array([1]), np.array([2]), 5, 1., 1.)


if __name__ == '__main__':
    unittest.main()
