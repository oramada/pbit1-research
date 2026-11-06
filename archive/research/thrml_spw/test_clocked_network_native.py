"""Same-path public-factor reconstruction and explicit clock partition controls."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from .clocked_network_native import ClockedPositiveNetwork
from .calibrated_mnist_path_audit import PublicLayer
from . import network_drift_clock as clock


class ClockedNetworkTests(unittest.TestCase):
    def test_segment_means_before_relu_and_native_counts(self):
        w=[np.array([[.01,.3],[.9,.7]],np.float64),np.array([[.4],[.8]],np.float64)]
        biases=[np.array([-.5,.1],np.float32),np.array([.2],np.float32)]
        net=ClockedPositiveNetwork(w,biases);layers=[PublicLayer(v) for v in w]
        x=np.array([[[1.,-.3],[.2,.8]]],np.float32)
        f0=np.array([[[-2.,.2,.9,-.8],[.7,-1.,-.2,1.]]],np.float32)
        f1=np.array([[[.2,-1.],[-.5,1.3]]],np.float32)
        base=jax.random.key(77);pre,logits=[np.asarray(v) for v in net.dataset_segments(8)(base,jnp.asarray(x),jnp.asarray(f0),jnp.asarray(f1))]
        kb=jax.random.fold_in(base,0);means=[];parts=[]
        for layer,fields in enumerate([f0[0],f1[0]]):
            keys=jax.vmap(lambda s:jax.random.fold_in(jax.random.fold_in(kb,layer),s))(jnp.arange(2))
            q=np.stack([np.asarray(layers[layer].frequencies(8)(keys[s:s+1],jnp.asarray(fields[s])))[0] for s in range(2)])
            if layer==0:
                pp=np.stack([x[0].astype(float)@a+biases[0] for a in q]);expect=pp.mean(0)
                np.testing.assert_allclose(pre[0],expect,atol=2e-6);hidden=np.maximum(pre[0],0)
                parts=pp
            else:
                expect=np.mean([hidden.astype(float)@a+biases[1] for a in q],axis=0)
                np.testing.assert_allclose(logits[0],expect,atol=2e-6)
        # Explicit negative control rejects segment-wise nonlinear averaging.
        self.assertGreater(np.max(np.abs(np.maximum(parts,0).mean(0)-np.maximum(parts.mean(0),0))),1e-3)
        fields=np.concatenate([f0[0],f1[0]],axis=1);key=jax.random.key(99)
        counts=np.asarray(net.segmented_counts(8)(key,jnp.asarray(fields)))
        public=PublicLayer(np.concatenate([a.reshape(-1) for a in w]))
        keys=jax.vmap(lambda s:jax.random.fold_in(key,s))(jnp.arange(2))
        for s in range(2):
            q=np.asarray(public.frequencies(8)(keys[s:s+1],jnp.asarray(fields[s])))[0]
            np.testing.assert_array_equal(counts[s],np.rint(q*8).astype(np.int32))

    def test_complete_serial_clock_partition(self):
        occupied=[]
        for k in range(clock.WINDOWS):occupied.extend(range(clock.calibration_start(k),clock.calibration_start(k)+clock.CAL_SEGMENTS))
        for b in range(clock.BATCHES):occupied.extend(range(clock.batch_start(b),clock.batch_start(b)+2*clock.LAYER_SEGMENTS))
        self.assertEqual(sorted(occupied),list(range(clock.TOTAL_SEGMENTS)))
        self.assertEqual(clock.TOTAL_SEGMENTS*clock.SEGMENT,681984)
        path=clock.standard_path(0,0,3);self.assertEqual(path.shape,(clock.TOTAL_SEGMENTS+1,3));np.testing.assert_array_equal(path[0],0)
        gain,offset=clock.static_device(0,0,3);idx=np.array([0,5,77]);command=np.array([-.5,.2,1.])
        fields=clock.field_schedule(gain,offset,path,idx,.04,command)
        np.testing.assert_array_equal(fields,(gain*command+offset+.04*path[idx]).astype(np.float32))


if __name__=='__main__':unittest.main()
