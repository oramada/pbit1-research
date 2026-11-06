import itertools,unittest
import numpy as np
from .probe_placement_costs import expected_recorded_changes


class ChangeTests(unittest.TestCase):
    def test_exact_iid_sequence_enumeration(self):
        for p in [0.,.2,.5,1.]:
            expected=0.
            for seq in itertools.product([0,1],repeat=4):
                probability=np.prod([p if b else 1-p for b in seq]);previous=0;changes=0
                for b in seq:changes+=b!=previous;previous=b
                expected+=probability*changes
            self.assertAlmostEqual(float(expected_recorded_changes(p,4)),expected,places=14)
        np.testing.assert_array_equal(expected_recorded_changes(np.array([0.,1.]),2048),[0.,1.])


if __name__=='__main__':unittest.main()
