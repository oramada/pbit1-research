import unittest
import numpy as np
from .probe_placement_math import weighted_moments,score_pair,design,decode


class ProbeTests(unittest.TestCase):
    def test_explicit_delta_and_inverse_fisher(self):
        codes=np.array([[30,110],[128,220],[170,60]],np.uint8);span=np.array([[.4,.9]],np.float32);m=weighted_moments(codes,span)
        b=decode(codes).astype(float);sp=np.repeat(span,128,0)[:3].astype(float);p=1/(1+np.exp(-2*b));s=(2*sp*p*(1-p))**2
        for low in [32,96,110,120]:
            high=255-low;x,y=decode(np.array([low,high],np.uint8)).astype(float);q=1/(1+np.exp(-2*np.array([x,y])));v=.25/(2048*q*(1-q))
            direct=float(np.mean(s*(v[0]*((y-b)/(y-x))**2+v[1]*((b-x)/(y-x))**2)))
            fisher=sum(4*2048*pr*(1-pr)*np.outer([z,1],[z,1]) for z,pr in zip([x,y],q));cov=np.linalg.inv(fisher)
            explicit=float(np.mean(s*(b*b*cov[0,0]+2*b*cov[0,1]+cov[1,1])))
            self.assertAlmostEqual(score_pair(m,low,high),direct,places=13);self.assertAlmostEqual(direct,explicit,places=13)
        d=design(codes,span);self.assertLessEqual(d['chosen']['nominal_first_order_weight_MSE'],d['fixed_baseline']['nominal_first_order_weight_MSE']);self.assertEqual(d['chosen']['low_code']+d['chosen']['high_code'],255)


if __name__=='__main__':unittest.main()
