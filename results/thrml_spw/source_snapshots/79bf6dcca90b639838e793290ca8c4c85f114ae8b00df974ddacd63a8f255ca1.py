import unittest
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference

class ConfirmationStatisticsChecks(unittest.TestCase):
    def test_paired_identical_losses_have_exact_zero_uncertainty(self):
        values=np.random.default_rng(8).normal(size=(5,128))
        np.testing.assert_array_equal(bootstrap_paired_difference(values,values),np.zeros(10000))

    def test_constant_effect_is_recovered_for_every_resample(self):
        baseline=np.random.default_rng(9).normal(size=(5,128))
        values=bootstrap_paired_difference(baseline+.25,baseline)
        np.testing.assert_allclose(values,.25,atol=1e-15)
        self.assertEqual(values.shape,(10000,))

if __name__=='__main__':unittest.main()
