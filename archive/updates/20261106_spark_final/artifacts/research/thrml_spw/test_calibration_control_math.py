import unittest
import numpy as np
from .calibration_control_math import estimate_command, decode_command


class ControlChecks(unittest.TestCase):
    def test_observed_high_probability_reduces_requested_bias_and_stays_finite(self):
        nominal = np.array([0., 0., 0.], np.float32)
        codes, offset = estimate_command(np.array([0, 50, 100]), 100, nominal, 0.)
        self.assertTrue(np.isfinite(offset).all())
        self.assertGreater(codes[0], codes[1]); self.assertGreater(codes[1], codes[2])
        np.testing.assert_array_equal(decode_command(codes), (-4+codes.astype(float)*8/255).astype(np.float32))

    def test_probe_uses_real_code_and_counts_are_validated(self):
        probe = decode_command(np.array(128, np.uint8))
        self.assertGreater(float(probe), 0.)
        with self.assertRaises(ValueError):
            estimate_command(np.array([11]), 10, np.array([0.]), probe)
        with self.assertRaises(ValueError):
            estimate_command(np.array([.5]), 10, np.array([0.]), probe)


if __name__ == '__main__':
    unittest.main()
