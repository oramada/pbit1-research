import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from thrml import FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .controlled_positive_native import ControlledPositiveNetwork


def public_states(bank,fields,key,t):
    program=FactorSamplingProgram(bank.program.gibbs_spec,[SpinGibbsConditional()],
        [SpinEBMFactor(bank.free_blocks,jnp.asarray(fields))],[])
    return np.asarray(sample_states(key,program,SamplingSchedule(1,t,1),
        [jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)],bank.clamp_state,[bank.readout])[0]).reshape(t,*bank.shape)


class FullNetworkChecks(unittest.TestCase):
    def test_batched_full_native_network_matches_public_paths_and_correct_activation_order(self):
        w=[np.array([[0.,.3],[.8,.6],[.4,1.]],np.float32),np.array([[.2,.7],[.5,.1]],np.float32)]
        b=[np.array([-.2,.15],np.float32),np.array([.1,-.1],np.float32)]
        network=ControlledPositiveNetwork(w,b);fields=(network.calibration_bank.biases+.17).astype(np.float32)
        x=np.array([[[.2,-.5,.7],[.4,.1,-.3]],[[-.1,.2,-.5],[.6,-.7,.1]]],np.float32)
        t=32;base=jax.random.key(1900);pre,logits=[np.asarray(a) for a in network.dataset(t)(base,jnp.asarray(x),jnp.asarray(fields))]
        negative_control_difference=0.
        for batch in range(2):
            k0,k1=jax.random.split(jax.random.fold_in(base,batch));split=network.free_split
            s0=public_states(network.layers[0].bank,fields[:split],k0,t)
            p=x[batch]@s0.mean(0)+b[0];h=np.maximum(p,0)
            s1=public_states(network.layers[1].bank,fields[split:],k1,t)
            y=h@s1.mean(0)+b[1]
            np.testing.assert_allclose(pre[batch],p,atol=3e-6)
            np.testing.assert_allclose(logits[batch],y,atol=3e-6)
            wrong=np.maximum(np.einsum('ni,tio->tno',x[batch],s0)+b[0],0).mean(0)
            negative_control_difference=max(negative_control_difference,float(np.max(np.abs(wrong-h))))
        self.assertGreater(negative_control_difference,.001)
        bank=network.calibration_bank;key=jax.random.key(1901)
        states=public_states(bank,fields,key,t).reshape(t,-1)
        np.testing.assert_array_equal(network.counts(t)(key,jnp.asarray(fields)),states[:,bank.free_indices].sum(0))
        self.assertEqual(sum(l.bank.logical_pbits for l in network.layers),10)


if __name__=='__main__':unittest.main()
