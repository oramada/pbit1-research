import unittest
import numpy as np
from .block_quantization_controls import quantize


class QuantizationControls(unittest.TestCase):
    def test_zero_is_preserved_and_local_groups_isolate_outliers(self):
        w=np.zeros((256,1),dtype=np.float32);w[0]=100;w[128]=.1;w[129]=-.1
        for method in ("symmetric","affine_zero"):
            global_q=quantize(w,4,method)
            local_q=quantize(w,4,method,128)
            np.testing.assert_array_equal(local_q[w==0],0)
            self.assertEqual(float(global_q[128,0]),0)
            self.assertGreater(float(local_q[128,0]),0)
            self.assertLess(float(local_q[129,0]),0)
        constant=np.full((128,2),.25,dtype=np.float32)
        for method in ("symmetric","affine_zero"):
            np.testing.assert_allclose(quantize(constant,4,method,128),constant,atol=1e-7)


if __name__ == "__main__":unittest.main()
