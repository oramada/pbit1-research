"""Diagnose a failed frequency tolerance by exact public F32 and F64-sum replay."""
import json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,FactorSamplingProgram,SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .calibrated_mnist_path_audit import PublicLayer
from .block_cached_control_worker import sha,atomic_json
OUT=Path('results/thrml_spw/all_blocks_clean')

class DualPrecisionObserver(AbstractObserver):
    readout:object
    x:jax.Array
    span:jax.Array
    shape:tuple
    def init(self):
        return (jnp.zeros(int(np.prod(self.shape)),jnp.int32),jnp.zeros((len(self.x),self.shape[1]),jnp.float32),jnp.zeros((len(self.x),self.shape[1]),jnp.float64))
    def __call__(self,program,state_free,state_clamped,carry,iteration):
        state=block_state_to_global(state_free+state_clamped,program.gibbs_spec);bits=from_global_state(state,program.gibbs_spec,[self.readout])[0];value=self.x@(bits.reshape(self.shape).astype(jnp.float32)*self.span)
        return (carry[0]+bits.astype(jnp.int32),carry[1]+value,carry[2]+value.astype(jnp.float64)),None

def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;p=json.loads((OUT/'protocol.json').read_text());stage=23;seed=3;c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;lo=np.repeat(a['lower'],128,0);sp=np.repeat(a['span'],128,0);field=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);xp=OUT/'audit_work/regenerated_inputs.npy';yp=OUT/'runs/seed3/stage23_native_output.npy';meta=json.loads(yp.with_suffix('.json').read_text());assert sha(xp)==meta['signature']['inputs_sha256'] and sha(yp)==meta['output_sha256'];x=np.load(xp);y=np.load(yp);assert x.shape==(32,65,3072);layer=PublicLayer(np.full(shape,.5));keys=[]
    for bank in range(4):
        key=jax.random.key(1170730)
        for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
        keys.append(key)
    q=np.asarray(layer.frequencies(1024)(jnp.stack(keys),jnp.asarray(field)));counts0=np.rint(q*1024).astype(np.uint16);del layer
    # Enable double only for the diagnostic accumulator. All fields, sampled
    # conditional probabilities, matmul inputs/products and naive sum stay F32.
    # Exact per-node count comparison below tests that the setting did not change
    # even one recorded random bit total at any of the 9,437,184 bank/coordinates.
    jax.config.update('jax_enable_x64',True);layer=PublicLayer(np.full(shape,.5));program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,jnp.asarray(field,dtype=jnp.float32))],[])
    def run(key,xx):
        observer=DualPrecisionObserver(layer.readout,xx,jnp.asarray(sp,dtype=jnp.float32),shape);carry,_=sample_with_observation(key,program,SamplingSchedule(1,1024,1),layer.initial,[],observer.init(),observer);count,total32,total64=carry;y32=total32/jnp.float32(1024)+xx@jnp.asarray(lo,dtype=jnp.float32)+jnp.asarray(a['bias'],dtype=jnp.float32)
        return count,y32,total64
    fn=jax.jit(run);counts=[];ys=[];high=[]
    for bank,key in enumerate(keys):
        xx=x[bank*8:(bank+1)*8].reshape(520,shape[0]);count,y32,total64=fn(key,jnp.asarray(xx,dtype=jnp.float32));count=np.asarray(count).reshape(shape);np.testing.assert_array_equal(count.astype(np.uint16),counts0[bank]);counts.append(count.astype(np.uint16));ys.append(np.asarray(y32).reshape(8,65,-1));high.append((np.asarray(total64)/1024+xx.astype(float)@lo.astype(float)+a['bias']).reshape(8,65,-1));print(f'Dual-precision witness bank{bank}: exact native integer counts',flush=True)
    replay=np.concatenate(ys);accurate=np.concatenate(high);expected=(x.reshape(4,520,shape[0]).astype(float)@(lo.astype(float)+sp.astype(float)*q.astype(float))+a['bias']).reshape(y.shape);scale=1+np.max(np.abs(expected));original_error=float(np.max(np.abs(y.astype(float)-expected))/scale);naive_error=float(np.max(np.abs(replay.astype(float)-y.astype(float)))/scale);accurate_error=float(np.max(np.abs(accurate-expected))/scale)
    # No relaxation of the old threshold: original still fails; accurate sum
    # must meet a substantially tighter diagnostic bound, and public F32 must
    # reproduce the existing implementation byte for byte.
    assert original_error>2e-5;np.testing.assert_array_equal(replay,y);assert accurate_error<1e-6,(accurate_error,naive_error)
    out=OUT/'rounding_witness';out.mkdir(exist_ok=True);np.savez_compressed(out/'arrays.npz',counts=counts0,replayed32=replay,high_precision_sum=accurate,float64_frequency_expected=expected)
    r={'passed':True,'seed':seed,'stage':stage,'inputs_sha256':sha(xp),'output_sha256':sha(yp),'controls_sha256':sha(c['path']),'original_fixed_frequency_tolerance':2e-5,'original_fixed_frequency_tolerance_passed':False,'original_scaled_frequency_error':original_error,'public_F32_output_bitwise_exact':True,'public_F32_scaled_error_vs_production':naive_error,'every_native_frequency_count_exact_with_x64_enabled':True,'bank_weight_coordinates_count_checked':int(counts0.size),'high_precision_sum_scaled_frequency_error':accurate_error,'high_precision_diagnostic_tolerance':1e-6,'arrays_sha256':sha(out/'arrays.npz'),'source_sha256':sha(__file__),'scope':'Same preserved weights/fields/inputs/keys, independently built public THRML factors. Counts match defaultF32 sampling at every coordinate; naiveF32 output is byte-identical to original. F64 sum of the same F32 per-sample dot products agrees much more closely with the F64 count reconstruction. Thus this specific old strict-tolerance failure is explained by F32 temporal accumulation. No original outputs, task metrics, quality rules or fixed tolerance changed; accurate-sum diagnostic is not a new deployed quality result.'};atomic_json(out/'audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
