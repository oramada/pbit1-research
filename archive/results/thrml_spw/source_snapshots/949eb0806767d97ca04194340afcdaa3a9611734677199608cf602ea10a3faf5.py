"""Validate full-cohort replay against every original bounded native array."""
from pathlib import Path
import json
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT,TRAIN,sha,write,key_bank
from results.thrml_spw.fashion_cnn_native.cohort.replay import recover
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled;path=ROOT/'cohort_selfcheck.json';assert not path.exists();training=json.loads((TRAIN/'manifest.json').read_text());audit=json.loads((ROOT/'preflight/evidence_audit.json').read_text());parent=json.loads((ROOT/'preflight/protocol.json').read_text());assert audit['passed'];x=np.load(TRAIN/'validation.npz')['inputs'][:8].astype(float).reshape(2,4,1,14,14);rows=[];updates=0
    for row in audit['rows']:
        mi,t=row['model'],row['T'];model=np.load(training['rows'][mi]['path']);key=jax.random.fold_in(jax.random.fold_in(jax.random.key(parent['key_root']),mi),t);keys=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)];raw=recover(x,model,t,keys);original=dict(np.load(row['path']));np.testing.assert_array_equal(receipt(raw),receipt(original))
        for name in raw:np.testing.assert_array_equal(raw[name],original[name])
        updates+=2*t*2664;rows.append({'model':mi,'T':t,'bank_arrays_exact':28});print(json.dumps(rows[-1]),flush=True)
    allkeys=[]
    for mi in range(5):
        for seed in range(10):
            for t in [64,256,1024]:
                for li in range(3):
                    whole=np.asarray(jax.random.key_data(key_bank(mi,seed,t,li,0,4)));parts=np.concatenate([np.asarray(jax.random.key_data(key_bank(mi,seed,t,li,0,2))),np.asarray(jax.random.key_data(key_bank(mi,seed,t,li,2,2)))]);np.testing.assert_array_equal(whole,parts);allkeys.extend(a.tobytes() for a in whole)
    assert len(allkeys)==1800 and len(set(allkeys))==1800 and updates==1918080
    write(path,{'passed':True,'rows':rows,'all420_arrays_byte_exact':True,'public_replay_updates':updates,'new_native_updates':0,'chunk_invariant_distinct_keys_checked':1800,'source_sha256':{str(f):sha(f) for f in [Path(__file__),ROOT/'cohort/replay.py',ROOT/'cohort/common.py']},'scope':'Validation-only complete causal recovery check; no test labels/quality scoring or native production. Key mapping is checked across all planned model/MC/budget/layer roots at first4banks.'})
if __name__=='__main__':main()
