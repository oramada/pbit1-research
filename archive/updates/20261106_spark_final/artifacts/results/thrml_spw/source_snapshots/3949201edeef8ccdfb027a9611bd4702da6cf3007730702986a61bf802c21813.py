import unittest
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_states
from .controlled_grouped_native import ControlledGroupedLinear

class ControlledGroupedChecks(unittest.TestCase):
    def test_actual_quantized_controls_and_same_path(self):
        w=np.array([[-.7,.2],[-.1,.3],[.2,.8],[.9,.9]],dtype=np.float32)
        b=np.array([.1,-.2],dtype=np.float32);x=np.array([[.2,-.5,.7,.4]],dtype=np.float32)
        layer=ControlledGroupedLinear(w,b,4,4.,4);bank=layer.bank
        self.assertLessEqual(len(np.unique(bank.biases)),16)
        self.assertGreater(np.max(np.abs(layer.realized_weights-w)),.02)
        self.assertEqual(layer.controls()['field_bits'],4)
        self.assertEqual(layer.controls()['metadata_bits'],128)
        key=jax.random.key(903);t=64
        states=np.asarray(sample_states(key,bank.program,SamplingSchedule(1,t,1),[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)],bank.clamp_state,[bank.readout])[0]).reshape(t,*w.shape)
        expected=x@(layer.lower+layer.span*states.mean(0))+b
        np.testing.assert_allclose(np.asarray(layer.compiled(t)(key,jnp.asarray(x))),expected,atol=2e-6)

if __name__=='__main__':unittest.main()
