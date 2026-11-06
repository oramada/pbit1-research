"""Same-default-precision public replay isolates temporal F32 summation error."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,FactorSamplingProgram,SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .calibrated_mnist_path_audit import PublicLayer
from .block_cached_control_worker import sha,atomic_json
OUT=Path('results/thrml_spw/all_blocks_clean')

class PlainObserver(AbstractObserver):
    readout:object
    x:jax.Array
    span:jax.Array
    shape:tuple
    def init(self):return (jnp.zeros(int(np.prod(self.shape)),jnp.int32),jnp.zeros((len(self.x),self.shape[1]),jnp.float32))
    def __call__(self,program,state_free,state_clamped,carry,iteration):
        state=block_state_to_global(state_free+state_clamped,program.gibbs_spec);bits=from_global_state(state,program.gibbs_spec,[self.readout])[0];value=self.x@(bits.reshape(self.shape).astype(jnp.float32)*self.span)
        return (carry[0]+bits.astype(jnp.int32),carry[1]+value),None

class CompensatedObserver(AbstractObserver):
    readout:object
    x:jax.Array
    span:jax.Array
    shape:tuple
    def init(self):
        return (jnp.zeros(int(np.prod(self.shape)),jnp.int32),jnp.zeros((len(self.x),self.shape[1]),jnp.float32),jnp.zeros((len(self.x),self.shape[1]),jnp.float32))
    def __call__(self,program,state_free,state_clamped,carry,iteration):
        state=block_state_to_global(state_free+state_clamped,program.gibbs_spec);bits=from_global_state(state,program.gibbs_spec,[self.readout])[0];value=self.x@(bits.reshape(self.shape).astype(jnp.float32)*self.span);delta=value-carry[2];total=jax.lax.optimization_barrier(carry[1]+delta);correction=(total-carry[1])-delta
        return (carry[0]+bits.astype(jnp.int32),total,correction),None

def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;p=json.loads((OUT/'protocol.json').read_text());stage=23;seed=3;c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;lo=np.repeat(a['lower'],128,0);sp=np.repeat(a['span'],128,0);field=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);xp=OUT/'audit_work/regenerated_inputs.npy';yp=OUT/'runs/seed3/stage23_native_output.npy';meta=json.loads(yp.with_suffix('.json').read_text());assert sha(xp)==meta['signature']['inputs_sha256'] and sha(yp)==meta['output_sha256'];x=np.load(xp);y=np.load(yp);assert x.shape==(32,65,3072);layer=PublicLayer(np.full(shape,.5));keys=[]
    for bank in range(4):
        key=jax.random.key(1170730)
        for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
        keys.append(key)
    q=np.asarray(layer.frequencies(1024)(jnp.stack(keys),jnp.asarray(field)));counts0=np.rint(q*1024).astype(np.uint16)
    def plain(key,xx,fields):
        program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,fields)],[])
        observer=PlainObserver(layer.readout,xx,jnp.asarray(sp),shape);carry,_=sample_with_observation(key,program,SamplingSchedule(1,1024,1),layer.initial,[],observer.init(),observer);return carry[0],carry[1]/1024+xx@jnp.asarray(lo)+jnp.asarray(a['bias'])
    def compensated(key,xx,fields):
        program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,fields)],[])
        observer=CompensatedObserver(layer.readout,xx,jnp.asarray(sp),shape);carry,_=sample_with_observation(key,program,SamplingSchedule(1,1024,1),layer.initial,[],observer.init(),observer);return carry
    fplain=jax.jit(plain);fcomp=jax.jit(compensated);ys=[];high=[]
    for bank,key in enumerate(keys):
        xx=x[bank*8:(bank+1)*8].reshape(520,shape[0]);count,yy=fplain(key,jnp.asarray(xx),jnp.asarray(field));np.testing.assert_array_equal(np.asarray(count).reshape(shape).astype(np.uint16),counts0[bank]);ys.append(np.asarray(yy).reshape(8,65,-1));count,total,correction=fcomp(key,jnp.asarray(xx),jnp.asarray(field));np.testing.assert_array_equal(np.asarray(count).reshape(shape).astype(np.uint16),counts0[bank]);precise=(np.asarray(total).astype(float)-np.asarray(correction).astype(float))/1024+xx.astype(float)@lo.astype(float)+a['bias'];high.append(precise.reshape(8,65,-1));print(f'F32/compensated witness bank{bank}: all counts exact',flush=True)
    replay=np.concatenate(ys);accurate=np.concatenate(high);expected=(x.reshape(4,520,shape[0]).astype(float)@(lo.astype(float)+sp.astype(float)*q.astype(float))+a['bias']).reshape(y.shape);scale=1+np.max(np.abs(expected));original_error=float(np.max(np.abs(y.astype(float)-expected))/scale);naive_error=float(np.max(np.abs(replay.astype(float)-y.astype(float)))/scale);accurate_error=float(np.max(np.abs(accurate-expected))/scale)
    assert original_error>2e-5;np.testing.assert_array_equal(replay,y);assert accurate_error<1e-6,(accurate_error,naive_error)
    out=OUT/'rounding_witness';out.mkdir(exist_ok=True);np.savez_compressed(out/'arrays.npz',counts=counts0,replayed32=replay,compensated_sum=accurate,float64_frequency_expected=expected)
    r={'passed':True,'seed':seed,'stage':stage,'inputs_sha256':sha(xp),'output_sha256':sha(yp),'controls_sha256':sha(c['path']),'original_fixed_frequency_tolerance':2e-5,'original_fixed_frequency_tolerance_passed':False,'original_scaled_frequency_error':original_error,'public_F32_output_bitwise_exact':True,'public_F32_scaled_error_vs_production':naive_error,'every_native_frequency_count_exact_in_both_public_replays':True,'bank_weight_coordinates_count_checked':int(counts0.size),'compensated_sum_scaled_frequency_error':accurate_error,'compensated_diagnostic_tolerance':1e-6,'arrays_sha256':sha(out/'arrays.npz'),'default_JAX_x64_enabled':False,'native_observed_updates_executed':12*1024*int(np.prod(shape)),'source_sha256':sha(__file__),'scope':'Same preserved controls/inputs/keys and defaultF32 sampling throughout; no global precision change. Independent public F32 sum output is byte-identical to original. KahanF32 sum with explicit optimization barrier, corrected inF64 onhost plusF64 center, agrees much more closely with F64 count reconstruction. Counts match every coordinate in both replays. This explains the original fixed-tolerance failure by F32 temporal accumulation. No production outputs, task metrics, quality rules or original tolerance changed; compensated result is only a numerical diagnostic.'};atomic_json(out/'audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
