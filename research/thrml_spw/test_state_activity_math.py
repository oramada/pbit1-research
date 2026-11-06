"""Exhaustive small-sequence verification of reset and retained flip moments."""
import itertools,unittest
import numpy as np
from .state_activity_math import moments
class MomentTests(unittest.TestCase):
    def test_exact_sequence_enumeration_both_initializations(self):
        for p in [0.,.01,.2,.5,.9,1.]:
            for t in range(1,7):
                for retained in [False,True]:
                    terms=[]
                    for bits in itertools.product([0,1],repeat=t+int(retained)):
                        probability=np.prod([p if b else 1-p for b in bits]);states=bits if retained else (0,)+bits;flips=sum(a!=b for a,b in zip(states,states[1:]));terms.append((probability,flips))
                    mean=sum(w*f for w,f in terms);var=sum(w*(f-mean)**2 for w,f in terms);actual=moments(p,t,retained);np.testing.assert_allclose(actual,[mean,var],rtol=2e-12,atol=2e-13)
    def test_stationary_rate_bound_and_endpoint_initialization(self):
        q=np.linspace(0,1,1001)
        for t in [1,4,16,64,256]:
            mean,var=moments(q,t,True);self.assertTrue(np.all(mean<=t/2+1e-12));self.assertTrue(np.all(var>=-1e-12));np.testing.assert_array_equal(moments(np.array([0.,1.]),t)[0],[0,1]);np.testing.assert_array_equal(moments(np.array([0.,1.]),t,True)[0],[0,0])
if __name__=='__main__':unittest.main()
