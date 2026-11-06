"""Public native forward and independent explicit STE derivative checks."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from .probability_training_native import TrainingNetwork,command,manual_ste_gradient,profile,adam
from .calibrated_mnist_path_audit import PublicLayer


class TrainingTests(unittest.TestCase):
    def test_public_paths_and_manual_backward(self):
        rng=np.random.default_rng(123);theta=rng.uniform(-5,1,2368).astype(np.float32)
        x=rng.normal(0,.2,(8,64)).astype(np.float32);y=np.arange(8,dtype=np.int32)
        biases=[np.linspace(-1,1,32,dtype=np.float32),np.linspace(-.5,.5,10,dtype=np.float32)]
        net=TrainingNetwork(biases,16);key=jax.random.key(61)
        for method in ['expected','native_ideal','native_device']:
            gain,offset=(np.ones(2368),np.zeros(2368)) if method!='native_device' else profile(0,0,'train')
            fields=(gain*command(theta).astype(float)+offset).astype(np.float32)
            (loss,(pre,logits)),grad=net.objective(method)(jnp.asarray(theta),jnp.asarray(x),jnp.asarray(y),key,jnp.asarray(fields))
            pre,logits,grad=np.asarray(pre),np.asarray(logits),np.asarray(grad)
            ce,g=manual_ste_gradient(theta,x,y,pre,logits)
            np.testing.assert_allclose(loss,ce,atol=2e-6);np.testing.assert_allclose(grad,g,atol=3e-7,rtol=2e-5)
            if method=='expected':continue
            k0,k1=jax.random.split(key)
            q0=np.asarray(PublicLayer(np.full((64,32),.5)).frequencies(16)(k0[None],jnp.asarray(fields[:2048])))[0]
            pp=x.astype(float)@q0.astype(float)+biases[0];np.testing.assert_allclose(pre,pp,atol=3e-6)
            q1=np.asarray(PublicLayer(np.full((32,10),.5)).frequencies(16)(k1[None],jnp.asarray(fields[2048:])))[0]
            ll=np.maximum(pre,0).astype(float)@q1.astype(float)+biases[1];np.testing.assert_allclose(logits,ll,atol=3e-6)

    def test_device_domains_and_optimizer_projection(self):
        for domain in ['train','validation','test']:
            a,d=profile(0,0,domain);self.assertTrue(np.all(a>0));np.testing.assert_array_equal(a,profile(0,0,domain)[0])
        self.assertFalse(np.array_equal(profile(0,0,'train')[0],profile(0,0,'test')[0]))
        theta=jnp.array([7.99,-7.99]);gradient=jnp.array([-1.,1.]);new,m,v=adam(theta,gradient,jnp.zeros(2),jnp.zeros(2),1,learning_rate=1.)
        np.testing.assert_array_equal(new,np.array([8.,-8.],np.float32))
        c=np.asarray(command(jnp.array([-8.,8.])));np.testing.assert_array_equal(c,[-4.,4.])


if __name__=='__main__':unittest.main()
