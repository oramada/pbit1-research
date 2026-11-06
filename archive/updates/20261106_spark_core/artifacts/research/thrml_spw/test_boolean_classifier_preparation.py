import unittest
import numpy as np
from .prepare_boolean_classifier import finite_field,budgets
class PreparationTests(unittest.TestCase):
    def test_budget_and_counter_width(self):
        b=budgets();self.assertEqual(b['uncompressed_required_payload_total'],1459500000)
        for a,c in zip(b['R1_samples'],b['R3_samples']):self.assertEqual(a*2368,c*3136);self.assertLess(a,65536);self.assertLess(64*c,2**32)
    def test_aggregate_counts_need_more_than_uint16(self):
        self.assertGreater(64*max(budgets()['R3_samples']),np.iinfo(np.uint16).max)
    def test_control_endpoints_and_invalid_probabilities(self):
        c,f=finite_field(np.array([0,.25,.5,.75,1]));np.testing.assert_array_equal(c[[0,-1]],[0,255]);np.testing.assert_array_equal(f[[0,-1]],[-4,4]);self.assertEqual(f.dtype,np.dtype('float32'));self.assertTrue(np.all(np.diff(f)>0))
        for x in [-.1,1.1,np.nan]:
            with self.assertRaises(ValueError):finite_field([x])
    def test_exact_affine_boolean_readout_with_negative_inputs(self):
        # Exhaust all states of two weight bits and two activation bits.
        for mask in range(16):
            bits=np.array([(mask>>i)&1 for i in range(4)]);w=bits[:2];a=bits[2:];low=-.42;span=3.25
            self.assertAlmostEqual(float((low+span*a)@w),float(low*w.sum()+span*np.bitwise_and(a,w).sum()),places=14)
if __name__=='__main__':unittest.main()
