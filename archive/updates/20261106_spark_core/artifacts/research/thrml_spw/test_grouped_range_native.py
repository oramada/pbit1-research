import unittest
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_states
from .grouped_range_native import GroupedRangeNativeLinear

class GroupedChecks(unittest.TestCase):
    def test_same_path_ragged_signed_constant_groups(self):
        w=np.array([[-.3,.5],[.1,.5],[.6,.5],[.4,.8],[.2,.1],[.3,-.2],[.7,.6]],dtype=np.float32)
        b=np.array([.1,-.2],dtype=np.float32)
        x=np.array([[.2,-.5,.7,.3,-.1,.6,.2],[-.2,.1,.3,.4,.7,-.2,.1]],dtype=np.float32)
        layer=GroupedRangeNativeLinear(w,b,3);bank=layer.bank
        np.testing.assert_allclose(layer.realized_weights,w,atol=4e-8)
        self.assertEqual(layer.controls()['groups'],6)
        self.assertEqual(layer.controls()['constant_groups'],3)
        self.assertEqual(layer.controls()['metadata_bits'],384)
        key=jax.random.key(721);t=32
        initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)]
        states=np.asarray(sample_states(key,bank.program,SamplingSchedule(1,t,1),initial,bank.clamp_state,[bank.readout])[0]).reshape(t,*w.shape)
        reference=x@(layer.lower+layer.span*states.mean(0))+b
        np.testing.assert_allclose(np.asarray(layer.compiled(t)(key,jnp.asarray(x))),reference,atol=2e-6)

    def test_singleton_groups_are_exact_clamps(self):
        w=np.array([[-2.,3.],[4.,-.5]],dtype=np.float32)
        layer=GroupedRangeNativeLinear(w,np.zeros(2),1)
        self.assertEqual(len(layer.bank.free_indices),0)
        x=jnp.array([[.2,-.7]])
        np.testing.assert_allclose(np.asarray(layer.compiled(4)(jax.random.key(9),x)),np.asarray(x)@w,atol=1e-6)

if __name__=='__main__':unittest.main()
