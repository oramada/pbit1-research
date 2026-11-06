import itertools,unittest
from .sample_allocation_math import allocate
class AllocationTest(unittest.TestCase):
    def test_exhaustive_optimum_and_ties(self):
        for a,units,budget in [([.2,.9,.4],[1,3,2],12),([1.,1.,1.],[1,1,1],7),([.1,.7,.8],[2,1,3],14)]:
            levels=[1,2,4];options=[(sum(c/t for c,t in zip(a,path)),path) for path in itertools.product(levels,repeat=3) if sum(u*t for u,t in zip(units,path))==budget];expected=min(options);actual=allocate(a,units,budget,levels);self.assertEqual(tuple(actual['levels']),expected[1]);self.assertAlmostEqual(actual['objective'],expected[0]);self.assertEqual(actual['budget_used'],budget)
    def test_infeasible(self):self.assertRaises(ValueError,allocate,[1.],[2],3,[1,2])
if __name__=='__main__':unittest.main()
