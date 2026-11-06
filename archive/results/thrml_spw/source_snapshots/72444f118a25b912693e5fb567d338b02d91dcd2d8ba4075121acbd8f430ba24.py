"""Independent public-THRML frequency replay for every all-block operator."""
import json,sys,traceback,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .calibrated_mnist_path_audit import PublicLayer
from .block_cached_control_worker import sha,atomic_json as save_json
OUT=Path('results/thrml_spw/all_blocks_clean')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class Worker:
    def __init__(self):self.p=json.loads((OUT/'protocol.json').read_text());self.layers={};self.fn={};self.results=[];self.updates=0
    def execute(self,r):
        stage,seed=int(r['stage']),int(r['seed']);c=self.p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;n=int(np.prod(shape));lo=np.repeat(a['lower'],128,0);sp=np.repeat(a['span'],128,0);b=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32)
        cache=tuple(shape)
        if cache not in self.layers:
            self.layers[cache]=PublicLayer(np.full(shape,.5));self.fn[cache]=self.layers[cache].frequencies(1024);assert self.layers[cache].n_free==n and not self.layers[cache].clamped
        folder=OUT/'runs'/f'seed{seed}';yp=folder/f'stage{stage}_native_output.npy';meta=json.loads(yp.with_suffix('.json').read_text());man=json.loads(yp.with_suffix('.manifest.json').read_text());assert man['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and man['packages']['thrml']=='0.1.4';xp=Path(r['regenerated_inputs']);assert sha(xp)==meta['signature']['inputs_sha256']==r['expected_input_sha256'] and sha(yp)==meta['output_sha256']
        sig=meta['signature'];assert sig['stage']==stage and sig['seed']==seed and sig['samples']==1024 and sig['contexts_per_bank']==8 and sig['controls_sha256']==c['sha256']
        for name,h in sig['source_sha256'].items():assert sha(name)==h
        assert meta['field_sha256']==digest(b);x,y=np.load(xp),np.load(yp);assert x.shape==(32,65,shape[0]) and y.shape==(32,65,shape[1]) and x.dtype==y.dtype==np.float32;keys=[]
        for bank in range(4):
            key=jax.random.key(1170730)
            for value in [seed,stage,bank]:key=jax.random.fold_in(key,value)
            keys.append(key)
        q=np.asarray(self.fn[cache](jnp.stack(keys),jnp.asarray(b))).astype(float);expected=(x.reshape(4,520,shape[0]).astype(float)@(lo+sp*q)+a['bias']).reshape(y.shape);error=float(np.max(np.abs(y.astype(float)-expected))/(1+np.max(np.abs(expected))))
        strict=error<2e-5;witness_path=None
        if not strict:
            # Preserve the original numerical failure; require a stronger exact
            # native/F32 witness, never a larger fitted tolerance.
            assert (seed,stage)==(3,23),(seed,stage,error)
            witness_path=OUT/'rounding_witness/audit.json';w=json.loads(witness_path.read_text());assert w['passed'] and w['public_F32_output_bitwise_exact'] and w['every_native_frequency_count_exact_in_both_public_replays'] and not w['original_fixed_frequency_tolerance_passed'] and not w['compensated_tight_tolerance_passed']
            assert w['inputs_sha256']==sha(xp) and w['output_sha256']==sha(yp) and w['controls_sha256']==sha(c['path']) and w['original_fixed_frequency_tolerance']==2e-5
            ar=OUT/'rounding_witness/arrays.npz';assert sha(ar)==w['arrays_sha256'];proof=np.load(ar);np.testing.assert_array_equal(proof['counts'],np.rint(q*1024).astype(np.uint16));np.testing.assert_array_equal(proof['replayed32'],y);np.testing.assert_array_equal(proof['float64_frequency_expected'],expected)
            assert np.all(np.abs(y.astype(float)-expected)<=proof['roundoff_bound']);np.testing.assert_allclose(error,w['original_scaled_frequency_error'],rtol=1e-12,atol=1e-15)
            assert abs(float(np.max(np.abs(proof['compensated_sum']-expected))/(1+np.max(np.abs(expected))))-w['compensated_sum_scaled_frequency_error'])<1e-15

        prob=(1/(1+np.exp(-2*b.astype(float)))).reshape(shape);mean=x.astype(float)@(lo+sp*prob)+a['bias'];pred=float(np.mean(np.mean(x.astype(float)**2,axis=(0,1))@(sp.astype(float)**2*prob*(1-prob)))/1024)
        np.testing.assert_allclose(meta['predicted_sampling_mse'],pred,atol=1e-12,rtol=1e-11);np.testing.assert_allclose(meta['sampling_mse_vs_programmed_expectation'],np.mean((y.astype(float)-mean)**2),atol=1e-12,rtol=1e-11)
        assert meta['free_nodes']==n and meta['clamped_nodes']==0 and meta['bank_batches']==4 and meta['free_node_updates']==meta['logical_weight_readouts']==4*1024*n and meta['binary_product_terms']==32*65*1024*n
        self.updates+=4*1024*n;self.results.append({'seed':seed,'stage':stage,'scaled_output_error':error,'fixed_frequency_tolerance_passed':bool(strict),'exact_public_F32_roundoff_witness':str(witness_path) if witness_path else None});return {'status':'ok','scaled_output_error':error}
    def finish(self):
        assert {(r['seed'],r['stage']) for r in self.results}=={(s,k) for s in range(5) for k in range(24)} and len(self.results)==120;assert self.updates==5*4*1024*42467328
        r={'passed':True,'scope_is_native_execution_verification_not_all_strict_tolerances_passing':True,'strict_frequency_audit_passed':all(v['fixed_frequency_tolerance_passed'] for v in self.results),'strict_frequency_failures_retained':[v for v in self.results if not v['fixed_frequency_tolerance_passed']],'operators':120,'all_native_nodes_replayed':True,'input_hashes_exact':120,'max_scaled_output_error':max(r['scaled_output_error'] for r in self.results),'tolerance':2e-5,'native_free_node_updates_checked':self.updates,'source_sha256':sha(__file__),'runs':self.results,'scope':'Independent public unary-factor/frequency reconstruction from exact regenerated model inputs. Same CUDA keys/fields; 119 operators meet the unchanged2e-5 frequency tolerance. The one preserved failure requires byte-identical independently built publicF32 execution, exact integer counts and a derived roundoff envelope; tighter compensation target remains failed. No quality data or tolerance changed. Not physical hardware.'};save_json(OUT/'native_path_audit.json',r);return {'status':'ok','operators':120}

def main():
    assert jax.default_backend()=='gpu';w=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(w.finish() if r.get('action')=='finish' else w.execute(r)),flush=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
