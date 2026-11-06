"""Statistical checks for the simulator adapter's sign, scaling, and endpoint policy."""
import unittest
import numpy as np

from .native import NativeProbabilityBank


class NativeBankChecks(unittest.TestCase):
    def test_stationary_probability_not_just_initialization(self):
        p=np.array([0.01,0.1,0.25,0.5,0.73,0.99])
        bank=NativeProbabilityBank(p)
        samples=bank.sample(730,4096,16)
        mean=samples.mean(axis=(0,1))
        tolerance=6*np.sqrt(p*(1-p)/(4096*16))
        self.assertTrue(np.all(np.abs(mean-p)<tolerance), (mean,p,tolerance))
        np.testing.assert_array_equal(samples,bank.sample(730,4096,16,initial_one=True))
        self.assertEqual(bank.logical_pbits,len(p))

    def test_beta_compensated_bias(self):
        bank=NativeProbabilityBank([0.1,0.73,0.9],beta=2)
        mean=bank.sample(73,4096,8).mean(axis=(0,1))
        np.testing.assert_allclose(mean,[.1,.73,.9],atol=.02)

    def test_exact_endpoint_policy_and_order(self):
        p=np.array([[1.,.73],[0.,1.]])
        samples=NativeProbabilityBank(p).sample(9,1024,4)
        self.assertEqual(samples.shape,(4,1024,2,2))
        self.assertTrue(samples[:,:,0,0].all())
        self.assertFalse(samples[:,:,1,0].any())
        self.assertTrue(samples[:,:,1,1].all())
        np.testing.assert_array_equal(NativeProbabilityBank([0.,1.]).sample(1,10)[0],
                                      np.tile([False,True],(10,1)))

    def test_invalid_probabilities(self):
        for p in ([-.1],[1.1],[np.nan],[]):
            with self.assertRaises(ValueError):NativeProbabilityBank(p)


if __name__=="__main__":unittest.main()
