"""GPU validation-only full word budgets with separate public raw recovery."""
import json,time,os
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.word_comparison.common import ROOT,OUT,TRAIN,sha,write,save,guard,parent_closed
from results.thrml_spw.fashion_cnn_native.word_reference.native import encode,forward
from results.thrml_spw.fashion_cnn_native.word_reference.witness import recover
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();dest=OUT/'GPU_preflight';dest.mkdir(exist_ok=True);assert not (dest/'audit.json').exists();assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';train=json.loads((TRAIN/'manifest.json').read_text());model=np.load(train['rows'][0]['path']);enc=encode(model);x=np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2,8,1,14,14);rows=[];native=0;public=0
    for t in p['word_samples']:
        key=jax.random.fold_in(jax.random.key(1597520),t);kk=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)];start=time.monotonic();cold=forward(x,model,enc,t,kk);cold_sec=time.monotonic()-start;start=time.monotonic();warm=forward(x,model,enc,t,kk);warm_sec=time.monotonic()-start;ref=recover(x,model,t,kk,enc);np.testing.assert_array_equal(receipt(cold),receipt(warm));np.testing.assert_array_equal(receipt(warm),receipt(ref))
        for name in warm:np.testing.assert_array_equal(warm[name],ref[name])
        path=dest/f'T{t}.npz';save(path,**warm);u=2*t*4*2664;native+=2*u;public+=u;rows.append({'T':t,'path':str(path),'sha256':sha(path),'cold_seconds':cold_sec,'warm_seconds':warm_sec,'all28_bank_arrays_byte_exact':True});print(json.dumps(rows[-1]),flush=True)
    assert native==14321664 and public==7160832;guard(p);write(dest/'audit.json',{'passed':True,'rows':rows,'native_updates':native,'public_updates':public,'scope':'Model0/16validation inputs, two8image banks, fixed matched budgets; no labels/scores. Cold+warm reuse trajectories, not new independent replications. Host timings include preparation/transfers/external readout and are not hardware latency or a reliable full-cohort forecast.'})
if __name__=='__main__':main()
