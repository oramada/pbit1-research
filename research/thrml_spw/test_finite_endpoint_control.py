import unittest
import jax
import jax.numpy as jnp
import numpy as np
from .finite_endpoint_control import ENCODINGS,encoded_weights,quantize,decode
from .controlled_positive_native import ControlledPositiveNetwork
from .calibrated_mnist_path_audit import PublicLayer


class FiniteEndpointChecks(unittest.TestCase):
    def test_range_precision_and_no_hidden_clamps_in_complete_native_network(self):
        original=[np.array([[0.,.3],[.8,.6],[.4,1.]]),np.array([[.2,.7],[.5,.1]])]
        bias=[np.array([-.2,.15],np.float32),np.array([.1,-.1],np.float32)]
        x=np.array([[[.2,-.5,.7],[.4,.1,-.3]]],np.float32)
        for encoding in ENCODINGS:
            b=encoding['bound'];bits=encoding['bits']
            np.testing.assert_array_equal(decode(quantize(np.array([-2*b,0.,2*b]),bits,b),bits,b)[[0,2]],[-b,b])
            codes=quantize(np.array([-b,b]),bits,b);self.assertEqual(codes.dtype,np.uint8 if bits==8 else np.uint16)
            w=encoded_weights(original,encoding);net=ControlledPositiveNetwork(w,bias)
            fields=decode(quantize(net.calibration_bank.biases,bits,b),bits,b)
            if not encoding['clamp_endpoints']:
                self.assertEqual(len(net.calibration_bank.clamped_indices),0)
                self.assertEqual(len(net.calibration_bank.free_indices),10)
                self.assertTrue(all(np.all((a>0)&(a<1)) for a in w))
            key=jax.random.key(2300);k0,k1=jax.random.split(jax.random.fold_in(key,0));t=32
            pre,logits=[np.asarray(a) for a in net.dataset(t)(key,jnp.asarray(x),jnp.asarray(fields))]
            public=[PublicLayer(a) for a in w];split=public[0].n_free
            q0=np.asarray(public[0].frequencies(t)(jax.random.split(k0,1),jnp.asarray(fields[:split])))
            # The public frequency function takes keys directly; preserve exact
            # production layer keys, rather than replacing them with split keys.
            q0=np.asarray(public[0].frequencies(t)(jnp.stack([k0]),jnp.asarray(fields[:split])))
            expected=x[0]@q0[0]+bias[0];np.testing.assert_allclose(pre[0],expected,atol=3e-6)
            q1=np.asarray(public[1].frequencies(t)(jnp.stack([k1]),jnp.asarray(fields[split:])))
            np.testing.assert_allclose(logits[0],np.maximum(pre[0],0)@q1[0]+bias[1],atol=3e-6)


if __name__=='__main__':unittest.main()
