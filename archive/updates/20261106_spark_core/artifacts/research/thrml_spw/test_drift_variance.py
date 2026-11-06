"""Exact nonstationary Bernoulli-operator moment check, separate from sampling."""
import itertools
import unittest
import numpy as np


class NonstationaryMomentChecks(unittest.TestCase):
    def test_conditional_path_variance_and_mse_by_exhaustive_enumeration(self):
        q=np.array([[.15,.8],[.75,.3]])
        x=np.array([.7,-.3]);lower=np.array([-.2,.1]);span=np.array([.4,.8]);bias=.2
        states=np.array(list(itertools.product([0.,1.],repeat=4))).reshape(-1,2,2)
        probability=np.prod(np.where(states==1,q,1-q),axis=(1,2))
        outputs=np.mean(np.sum(x*(lower+span*states),axis=2),axis=1)+bias
        mean=float(np.sum(probability*outputs));variance=float(np.sum(probability*(outputs-mean)**2))
        target=float(np.sum(x*(lower+span*np.array([.3,.7])))+bias)
        formula_mean=float(np.sum(x*(lower+span*q.mean(0)))+bias)
        formula_variance=float(np.sum(x*x*span*span*np.mean(q*(1-q),axis=0))/2)
        wrong_stationary=float(np.sum(x*x*span*span*q.mean(0)*(1-q.mean(0)))/2)
        self.assertAlmostEqual(probability.sum(),1.)
        self.assertAlmostEqual(mean,formula_mean,places=14)
        self.assertAlmostEqual(variance,formula_variance,places=14)
        self.assertGreater(abs(variance-wrong_stationary),.001)
        self.assertAlmostEqual(float(np.sum(probability*(outputs-target)**2)),(formula_mean-target)**2+formula_variance,places=14)


if __name__=='__main__':unittest.main()
