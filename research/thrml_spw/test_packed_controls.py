import tempfile,unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .packed_controls import save_controls,load_controls
from .packed_native import PackedNativeLinear
from .controlled_grouped_native import ControlledGroupedLinear

class PackedControlChecks(unittest.TestCase):
    def test_control_fields_and_native_outputs_roundtrip_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'controls.npz'
            w=np.array([[-.3,.5],[.1,.5],[.6,.5],[.4,.8],[.2,.1],[.3,-.2],[.7,.6]],dtype=np.float32);bias=np.array([.1,-.2],dtype=np.float32)
            ledger=save_controls(path,w,bias,3);original=ControlledGroupedLinear(w,bias,8,4.,3);restored=PackedNativeLinear(path)
            np.testing.assert_array_equal(original.bank.biases,restored.bank.biases);np.testing.assert_array_equal(original.bank.free_indices,restored.bank.free_indices)
            np.testing.assert_array_equal(original.realized_weights,restored.realized_weights)
            self.assertEqual(load_controls(path)['modes'].size,w.size)
            x=jnp.array([[.2,-.5,.7,.3,-.1,.6,.2]])
            for t in [1,32]:np.testing.assert_array_equal(np.asarray(original.compiled(t)(jax.random.key(805),x)),np.asarray(restored.compiled(t)(jax.random.key(805),x)))
            self.assertGreater(ledger['archive_bytes'],ledger['numeric_payload_bytes'])
    def test_constant_and_singleton_groups_preserve_clamped_values(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'singleton.npz';w=np.array([[-2.,3.],[4.,-.5]],dtype=np.float32)
            save_controls(p,w,np.zeros(2),1);restored=PackedNativeLinear(p);self.assertEqual(len(restored.bank.free_indices),0)
            x=jnp.array([[.2,-.7]])
            np.testing.assert_allclose(np.asarray(restored.compiled(4)(jax.random.key(9),x)),np.asarray(x)@w,atol=1e-6)

if __name__=='__main__':unittest.main()
