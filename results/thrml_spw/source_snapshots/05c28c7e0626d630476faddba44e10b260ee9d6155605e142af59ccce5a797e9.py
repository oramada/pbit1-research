"""Guard resource accounting and independently packed public-state AND counting."""
import unittest
import numpy as np
from .boolean_classifier_common import cases
from .verify_boolean_classifier_study import packed_and_counts
class CohortTests(unittest.TestCase):
    def test_all_prespecified_cases_and_budgets(self):
        rows=cases({'samples':{'R1':[98,392,1568],'R3':[74,296,1184]}})
        self.assertEqual(len(rows),300);self.assertEqual(len({r['name'] for r in rows}),300)
        sums={r:sum(1250*c['T']*(2368 if r=='R1' else 3136) for c in rows if c['regime']==r) for r in ['R1','R3']}
        self.assertEqual(sums,{'R1':304584000000,'R3':304584000000})
    def test_time_packing_padding_and_shared_activations(self):
        rng=np.random.default_rng(29)
        for t in [1,7,8,9,74,1184]:
            w=rng.integers(0,2,(2,t,5,3),dtype=np.uint8).astype(bool);a=rng.integers(0,2,(2,t,8,5),dtype=np.uint8).astype(bool)
            wanted=np.einsum('ctbi,ctio->cbo',a.astype(np.uint32),w.astype(np.uint32),dtype=np.uint32,optimize=False)
            np.testing.assert_array_equal(packed_and_counts(w,a),wanted)
    def test_worst_case_aggregate_width(self):
        w=np.ones((1,1184,64,2),bool);a=np.ones((1,1184,8,64),bool);got=packed_and_counts(w,a)
        self.assertTrue(np.all(got==75776));self.assertEqual(got.dtype,np.dtype('uint32'))
if __name__=='__main__':unittest.main()
