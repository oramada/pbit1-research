import unittest
import numpy as np
from .transformer_floor_math import fields,moments


class FloorTests(unittest.TestCase):
    def test_known_inverse_and_bounded_endpoints(self):
        nominal=np.array([-4.,.015686275,4.],np.float32);a=np.array([.5,1.,2.]);d=np.array([0.,.1,-.1]);codes=np.array([0,127,255],np.uint8)
        out,oracle,inverse=fields(nominal,a,d,codes)
        self.assertEqual(inverse[0],-8.);self.assertEqual(out['oracle_continuous_bound4'][0],-2.)
        np.testing.assert_array_equal(out['oracle_continuous_bound4'][1:],nominal[1:])
        self.assertEqual(oracle[0],0)
        expected=(a*(-4+oracle.astype(float)*8/255).astype(np.float32).astype(float)+d).astype(np.float32)
        np.testing.assert_array_equal(out['oracle8'],expected)
        q={k:1/(1+np.exp(-2*v.astype(float))) for k,v in out.items()};r=moments(q,np.array([.2,.5,1.]))
        self.assertLess(abs(r['SSE_closure_error']),1e-15);self.assertGreater(r['oracle_range_SSE'],0)
        self.assertAlmostEqual(sum(v for k,v in r.items() if k not in ['weights','total_SSE','SSE_closure_error']),r['total_SSE'],places=14)


if __name__=='__main__':unittest.main()
