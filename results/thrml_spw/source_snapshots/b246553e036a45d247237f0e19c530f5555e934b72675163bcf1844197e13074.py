import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from thrml import FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .drift_clock import brownian_path,query_start,calibration_start,TOTAL_SEGMENTS,CALIBRATION_SEGMENTS
from .replica_calibration_native import ReplicaCalibrationLinear
from .packed_controls import save_controls


class DriftClockChecks(unittest.TestCase):
    def test_clock_and_chunkwise_independent_drift_reconstruction(self):
        self.assertEqual([query_start(a) for a in [0,8,32]],[64,128,320])
        self.assertEqual([calibration_start(a) for a in [0,8,32]],[0,64,256])
        self.assertEqual(TOTAL_SEGMENTS,328)
        for age in [0,8,32]:self.assertEqual(query_start(age)-calibration_start(age),CALIBRATION_SEGMENTS)
        path=brownian_path(3,7);rng=np.random.default_rng(2026103003);current=np.zeros(7)
        for i in range(TOTAL_SEGMENTS):
            current=current+rng.standard_normal(7)
            np.testing.assert_array_equal(path[i+1],current)

    def test_native_segment_counts_and_pre_activation_means_match_public_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'control.npz';w=np.array([[0.,.4],[.2,.6],[.7,.9],[.5,.1],[.3,.8]],np.float32)
            bias=np.array([.1,-.2],np.float32);save_controls(p,w,bias,3)
            layer=ReplicaCalibrationLinear(p,1);bank=layer.bank;x=jnp.array([[.2,-.5,.7,.3,-.1]])
            all_bits=[];outputs=[];count_sum=np.zeros(len(bank.free_indices),dtype=np.int32)
            for segment,shift in enumerate([-.7,.1,.8]):
                fields=np.linspace(-.3,.3,len(bank.free_indices),dtype=np.float32)+np.float32(shift)
                program=FactorSamplingProgram(bank.program.gibbs_spec,[SpinGibbsConditional()],
                    [SpinEBMFactor(bank.free_blocks,jnp.asarray(fields))],[])
                key=jax.random.key(1500+segment)
                bits=np.asarray(sample_states(key,program,SamplingSchedule(1,4,1),
                    [jnp.zeros(len(bank.free_indices),dtype=jnp.bool_)],bank.clamp_state,[bank.readout])[0]).reshape(4,-1)
                counts=np.asarray(layer.compiled_counts(4)(key,jnp.asarray(fields)))
                np.testing.assert_array_equal(counts,bits[:,bank.free_indices].sum(0));count_sum+=counts
                all_bits.append(bits);outputs.append(np.asarray(layer.compiled_with_fields(4)(key,x,jnp.asarray(fields))))
            trajectory=np.concatenate(all_bits)
            np.testing.assert_array_equal(count_sum,trajectory[:,bank.free_indices].sum(0))
            expected=np.asarray(x)@(layer.lower+layer.span*trajectory.mean(0).reshape(w.shape))+bias
            np.testing.assert_allclose(np.mean(outputs,axis=0),expected,atol=3e-6)


if __name__=='__main__':unittest.main()
