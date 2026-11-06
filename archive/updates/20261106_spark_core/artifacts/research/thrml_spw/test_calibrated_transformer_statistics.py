import unittest
import numpy as np
from .calibrated_transformer_statistics import summarize


class Tests(unittest.TestCase):
    def test_fixed64_shape_quality_rule(self):
        base=np.full((64,64),3.,float);tokens=np.zeros((64,65),np.int64);pred=np.zeros((5,64,64),np.int64)
        r,d=summarize(np.broadcast_to(base+.01,pred.shape),pred,base,pred[0],tokens)
        self.assertTrue(r['passes']);np.testing.assert_allclose(d['CE_difference'],.01,atol=1e-15)
        r,d=summarize(np.broadcast_to(base+.1,pred.shape),pred,base,pred[0],tokens);self.assertFalse(r['passes'])
        with self.assertRaises(ValueError):summarize(np.zeros((5,128,64)),np.zeros((5,128,64)),base,pred[0],tokens)


if __name__=='__main__':unittest.main()
