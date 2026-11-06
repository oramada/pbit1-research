"""Validate constant-range handling and exact-zero controls before scoring."""
import unittest
import numpy as np
from .fine_range_quantization_math import uniform
class QuantizerTests(unittest.TestCase):
    def test_nonzero_constant_groups_remain_constant(self):
        w=np.tile(np.array([.25,-.5,0],np.float32),(32,1))
        for bits in [2,4,8]:
            for method in ['minmax','affine_zero','symmetric']:
                q,_=uniform(w,bits,16,method);np.testing.assert_allclose(q,w,atol=1e-7,rtol=1e-7)
    def test_zero_preserving_lattices_and_signed_levels(self):
        w=np.array([-1,0,1,0]*4,np.float32)[:,None]
        for method in ['affine_zero','symmetric']:
            q,_=uniform(w,2,16,method);np.testing.assert_array_equal(q[w==0],np.zeros(np.count_nonzero(w==0),np.float32))
        q,_=uniform(w,2,16,'symmetric');np.testing.assert_array_equal(q,w)
    def test_minmax_four_exact_levels(self):
        w=np.tile(np.array([[-.75],[-.25],[.25],[.75]],np.float32),(4,1));q,m=uniform(w,2,16,'minmax');np.testing.assert_array_equal(q,w);self.assertEqual(m['code_bits'],32);self.assertEqual(m['range_metadata_bits'],64)
if __name__=='__main__':unittest.main()
