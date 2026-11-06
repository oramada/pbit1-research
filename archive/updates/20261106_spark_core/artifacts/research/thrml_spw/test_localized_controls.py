import unittest
import numpy as np
from .localized_controls import build,ideal_variance,decode,expand
class LocalizedControlsTest(unittest.TestCase):
    def test_zero_radius_is_finite_exact_constant(self):
        w=np.full((128,3),-.125,np.float32)
        for g in [16,32,64,128]:
            a=build(w,np.zeros(3,np.float32),g);lo,span=expand(a,g);q=1/(1+np.exp(-2*decode(a['codes']).astype(float)));np.testing.assert_array_equal(lo+span*q,w);np.testing.assert_array_equal(span,np.zeros_like(w));self.assertTrue(np.isfinite(q).all())
    def test_nested_variance_for_exact_representable_weights(self):
        w=np.arange(128*3,dtype=np.float64).reshape(128,3)/512;previous=ideal_variance(w,128)
        for g in [64,32,16]:
            current=ideal_variance(w,g);self.assertTrue(np.all(current<=previous));previous=current
if __name__=='__main__':unittest.main()
