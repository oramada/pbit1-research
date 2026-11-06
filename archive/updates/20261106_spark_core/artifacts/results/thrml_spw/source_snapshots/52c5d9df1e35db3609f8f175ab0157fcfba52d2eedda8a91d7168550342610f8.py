"""All-free packed affine operator and count observer vs public THRML paths."""
import tempfile,unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_transformer_native import export_controls,CalibratedGroupedLinear,decode,device
from .calibrated_mnist_path_audit import PublicLayer


class NativeTests(unittest.TestCase):
    def test_all_free_operator_and_calibration_counts(self):
        w=np.arange(24,dtype=np.float32).reshape(8,3)/9-.9;bias=np.array([.2,-.1,.3],np.float32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'controls.npz';export_controls(p,w,bias);layer=CalibratedGroupedLinear(p)
            self.assertEqual(len(layer.bank.free_indices),24);self.assertEqual(len(layer.bank.clamped_indices),0)
            a,o=device(0,0,24);fields=(a*layer.nominal.astype(float)+o).astype(np.float32)
            x=np.array([[.2,.5,-.1,.6,-.3,.7,-.8,.1]],np.float32);key=jax.random.key(41)
            y=np.asarray(layer.forward(16)(key,jnp.asarray(x),jnp.asarray(fields)))
            public=PublicLayer(np.full(w.shape,.5));q=np.asarray(public.frequencies(16)(key[None],jnp.asarray(fields)))[0]
            expected=x.astype(float)@(layer.lower+layer.span*q.astype(float))+bias
            np.testing.assert_allclose(y,expected,atol=3e-6)
            counts=np.asarray(layer.counts(16)(key,jnp.asarray(fields)))
            np.testing.assert_array_equal(counts,np.rint(q.reshape(-1)*16).astype(np.int32))
            np.testing.assert_array_equal(decode(np.array([0,255],np.uint8)),[-4.,4.])
            self.assertGreater(np.min(1/(1+np.exp(-2*layer.nominal.astype(float)))),0.)


if __name__=='__main__':unittest.main()
