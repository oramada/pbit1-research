import tempfile,unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_states
from .packed_controls import save_controls
from .replicated_native import ReplicatedNativeLinear

class ReplicaChecks(unittest.TestCase):
    def test_same_native_trajectory_matches_replica_weight_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'controls.npz';w=np.array([[-.3,.5],[.1,.5],[.6,-.2],[.4,.8],[.2,.1]],dtype=np.float32);b=np.array([.1,-.2],dtype=np.float32)
            save_controls(p,w,b,3);layer=ReplicatedNativeLinear(p,4);bank=layer.bank;t=32;key=jax.random.key(714)
            x=jnp.array([[.2,-.5,.7,.3,-.1]]);initial=[jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)]
            bits=np.asarray(sample_states(key,bank.program,SamplingSchedule(1,t,1),initial,bank.clamp_state,[bank.readout])[0]).reshape(t,4,*w.shape)
            expected=np.asarray(x)@(layer.lower+layer.span*bits.mean(axis=(0,1)))+b
            np.testing.assert_allclose(np.asarray(layer.compiled(t)(key,x)),expected,atol=2e-6)
            self.assertEqual(bank.logical_pbits,4*w.size);self.assertEqual(bank.shape,(4,*w.shape))

if __name__=='__main__':unittest.main()
