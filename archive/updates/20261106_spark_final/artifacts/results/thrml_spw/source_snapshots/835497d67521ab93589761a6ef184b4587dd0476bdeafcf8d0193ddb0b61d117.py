import unittest
import numpy as np
from .conservative_selection import summarize_profile

class ConservativeSelectionChecks(unittest.TestCase):
    def arrays(self):
        return np.full((128,64),2.),np.zeros((128,64),dtype=np.int64),np.zeros((128,65),dtype=np.int64)
    def test_fixed_small_effect_passes_both_constraints(self):
        ref,pred,tokens=self.arrays();native=np.broadcast_to(ref+.02,(5,128,64))
        result,_=summarize_profile(native,np.broadcast_to(pred,native.shape),ref,pred,tokens,1024)
        self.assertTrue(result['passes']);self.assertAlmostEqual(result['bootstrap_CE_upper'],.02)
    def test_favorable_mean_does_not_hide_seed_uncertainty(self):
        ref,pred,tokens=self.arrays();native=np.broadcast_to(ref,(5,128,64)).copy();native[3:]+=.09
        result,_=summarize_profile(native,np.broadcast_to(pred,native.shape),ref,pred,tokens,1024)
        self.assertTrue(result['point_tolerance_passed']);self.assertFalse(result['passes']);self.assertGreater(result['bootstrap_CE_upper'],np.log(1.05))
    def test_bank_shared_error_can_fail_despite_passing_mean_accuracy(self):
        ref,pred,tokens=self.arrays();native=np.broadcast_to(ref+.02,(5,128,64));sampled=np.broadcast_to(pred,native.shape).copy();sampled[:,0]=1
        result,_=summarize_profile(native,sampled,ref,pred,tokens,1024)
        self.assertTrue(result['point_tolerance_passed']);self.assertFalse(result['passes']);self.assertLess(result['bootstrap_accuracy_difference_lower'],-.01)

if __name__=='__main__':unittest.main()
