import unittest
import jax
import jax.numpy as jnp
import numpy as np
from .grouped_partial_native import PartialGroupedLinear
from .controlled_grouped_native import ControlledGroupedLinear

class PartialChecks(unittest.TestCase):
    def test_same_path_direct_execution_including_ragged_and_constant_groups(self):
        w=np.array([[-.3,.5],[.1,.5],[.6,.5],[.4,.8],[.2,.1],[.3,-.2],[.7,.6]],dtype=np.float32)
        b=np.array([.1,-.2],dtype=np.float32);x=np.array([[.2,-.5,.7,.3,-.1,.6,.2],[-.2,.1,.3,.4,.7,-.2,.1]],dtype=np.float32)
        direct=ControlledGroupedLinear(w,b,6,4.,3);partial=PartialGroupedLinear(w,b,6,4.,3)
        np.testing.assert_array_equal(direct.bank.biases,partial.bank.biases)
        for t in [1,32]:
            key=jax.random.key(706)
            np.testing.assert_allclose(np.asarray(partial.compiled(t)(key,jnp.asarray(x))),np.asarray(direct.compiled(t)(key,jnp.asarray(x))),atol=2e-6)
        ledger=partial.operation_ledger(2,32)
        self.assertEqual(ledger['binary_product_terms'],2*w.size*32)
        self.assertEqual(ledger['partial_output_accumulator_float32_values'],12)
        self.assertEqual(ledger['direct_output_accumulator_float32_values'],4)

    def test_all_clamped_singleton_groups_remain_exact(self):
        w=np.array([[-2.,3.],[4.,-.5]],dtype=np.float32);layer=PartialGroupedLinear(w,np.zeros(2),4,4.,1)
        x=jnp.array([[.2,-.7]])
        np.testing.assert_allclose(np.asarray(layer.compiled(4)(jax.random.key(9),x)),np.asarray(x)@w,atol=1e-6)

if __name__=='__main__':unittest.main()
