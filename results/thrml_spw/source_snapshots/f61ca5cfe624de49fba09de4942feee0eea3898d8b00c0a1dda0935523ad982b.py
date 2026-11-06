"""Validation-only full-budget GPU throughput and exact public replay."""
from pathlib import Path
import hashlib,json,datetime,time,os
import numpy as np
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'throughput_evidence'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def guard(p):
    for file,h in p['source_sha256'].items():assert sha(file)==h,file
    for file,h in p['input_artifact_sha256'].items():assert sha(file)==h,file
def main():
    import jax
    from importlib.metadata import version
    from results.thrml_spw.fashion_cnn_native.inference.native import encode
    from results.thrml_spw.fashion_cnn_native.inference.fast_native import forward
    from results.thrml_spw.fashion_cnn_native.compact.recover import recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    p=json.loads((OUT/'protocol.json').read_text());guard(p);assert not (OUT/'evidence_audit.json').exists()
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0'
    train=json.loads((ROOT/'training/manifest.json').read_text());model=np.load(train['rows'][0]['path']);x=np.load(ROOT/'training/validation.npz')['inputs'][:16].astype(float).reshape(2,8,1,14,14);enc=encode(model);rows=[];native_updates=0;public_updates=0
    for t in p['samples']:
        key=jax.random.fold_in(jax.random.key(p['key_root']),t);keys=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)]
        start=time.monotonic();cold=forward(x,model,enc,t,keys);cold_seconds=time.monotonic()-start
        start=time.monotonic();warm=forward(x,model,enc,t,keys);warm_seconds=time.monotonic()-start
        np.testing.assert_array_equal(receipt(cold),receipt(warm))
        start=time.monotonic();reference=recover(x,model,t,keys);public_seconds=time.monotonic()-start
        np.testing.assert_array_equal(receipt(warm),receipt(reference))
        for name in warm:np.testing.assert_array_equal(warm[name],reference[name])
        path=OUT/f'T{t}.npz';np.savez_compressed(path,**warm,receipts=receipt(warm));u=2*t*2664;native_updates+=2*u;public_updates+=u
        rows.append({'T':t,'path':str(path),'sha256':sha(path),'cold_native_seconds':cold_seconds,'warm_native_seconds':warm_seconds,'public_recovery_seconds':public_seconds,'cold_plus_warm_native_updates':2*u,'public_updates':u,'all28_bank_arrays_byte_exact':True,'minimum_8bit_gate_counter_bound':255*144*t,'scheduled_nodes_per_round':2664,'activation_pbits':0})
        write(OUT/'progress.json',{'rows':rows,'native_updates':native_updates,'public_updates':public_updates});print(json.dumps(rows[-1]),flush=True)
    assert native_updates==14321664 and public_updates==7160832;guard(p)
    result={'passed':True,'rows':rows,'native_updates':native_updates,'public_updates':public_updates,'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy']},'backend':'gpu','XLA_FLAGS':os.environ['XLA_FLAGS'],'protocol_sha256':sha(OUT/'protocol.json'),'scope':p['scope'],'performance_limit':'Host wall time includes vectorized patch preparation, transfers, sampling and external F64 readout. Cold includes compilation. Single warm repeat/two banks is a diagnostic, not a reliable full-cohort runtime prediction or physical hardware timing.'}
    write(OUT/'evidence_audit.json',result)
    paths=[OUT/'protocol.json',OUT/'evidence_audit.json',OUT/'progress.json']+[Path(r['path']) for r in rows]
    write(OUT/'completion.json',{'status':'full_budget_GPU_CNN_validation_replay_audited_goal_active','artifact_sha256':{str(f):sha(f) for f in paths}})
    print('Full-budget GPU CNN validation replay passes; no test scores',flush=True)
if __name__=='__main__':main()
