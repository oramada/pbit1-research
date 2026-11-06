import unittest
import numpy as np

from .control_bank import ControlledProbabilityBank
from .native import NativeProbabilityBank


class ControlBankChecks(unittest.TestCase):
    def test_actual_sampler_uses_quantized_fields(self):
        bank=ControlledProbabilityBank([.1,.5,.73,.9],4,4.)
        target=bank.realized_probabilities
        measured=bank.sample(12,4096,16).mean(axis=(0,1))
        np.testing.assert_allclose(measured,target,atol=.01)
        self.assertGreater(abs(target[1]-.5),.1)
        self.assertTrue(np.all(bank.codes<16))

    def test_endpoint_policy_and_no_original_mutation(self):
        p=np.array([0.,1e-8,.73,1.])
        original=NativeProbabilityBank(p)
        controlled=ControlledProbabilityBank(p,6,2.)
        self.assertEqual(controlled.logical_pbits,len(p))
        self.assertEqual(controlled.clipped_fields,1)
        self.assertAlmostEqual(original.realized_probabilities[1],p[1],places=12)
        self.assertGreater(controlled.realized_probabilities[1],.01)
        bits=controlled.sample(1,128,2)
        self.assertFalse(bits[:,:,0].any());self.assertTrue(bits[:,:,-1].all())


if __name__=="__main__":unittest.main()
