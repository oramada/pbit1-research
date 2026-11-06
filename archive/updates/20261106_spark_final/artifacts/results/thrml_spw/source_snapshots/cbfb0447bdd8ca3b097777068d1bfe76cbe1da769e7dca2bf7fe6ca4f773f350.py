"""Actual native execution verifies vectorized preparation on every prior case."""
from pathlib import Path
import hashlib,json,datetime,time
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.inference.native import encode
from results.thrml_spw.fashion_cnn_native.inference.fast_native import forward
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'fast_witness'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    parent=json.loads((ROOT/'preflight/protocol.json').read_text());audit=json.loads((ROOT/'preflight/evidence_audit.json').read_text());compact=json.loads((ROOT/'compact_witness/evidence_audit.json').read_text());assert audit['passed'] and compact['passed']
    training=json.loads((ROOT/'training/manifest.json').read_text());sources=dict(parent['source_sha256']);sources.update({str(f):sha(f) for f in [Path(__file__),ROOT/'inference/fast_native.py',ROOT/'compact/receipts.py']})
    p={'source_sha256':sources,'parent_audit_sha256':sha(ROOT/'preflight/evidence_audit.json'),'compact_audit_sha256':sha(ROOT/'compact_witness/evidence_audit.json'),'scope':'Actual native observer rerun of all15 fixed validation cases with vectorized patch preparation; exact arrays/receipts against independent-public-checked original cases. Same trajectories, not extra independent accuracy replication. No GPU throughput/test scores.'}
    (OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');x=np.load(ROOT/'training/validation.npz')['inputs'][:8].astype(float).reshape(2,4,1,14,14);rows=[];updates=0;start=time.monotonic()
    for r in audit['rows']:
        mi,t=r['model'],r['T'];model=np.load(training['rows'][mi]['path']);enc=encode(model);key=jax.random.fold_in(jax.random.fold_in(jax.random.key(parent['key_root']),mi),t);keys=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)]
        raw=forward(x,model,enc,t,keys);original=dict(np.load(r['path']));np.testing.assert_array_equal(receipt(raw),receipt(original))
        for name in raw:np.testing.assert_array_equal(raw[name],original[name])
        updates+=2*t*2664;rows.append({'model':mi,'T':t,'arrays_exact':28,'native_updates':2*t*2664});print(json.dumps(rows[-1]),flush=True)
    assert updates==1918080
    for f,h in sources.items():assert sha(f)==h
    result={'passed':True,'rows':rows,'all420_bank_arrays_byte_exact':True,'native_updates':updates,'additional_public_replay_updates':0,'elapsed_seconds':time.monotonic()-start,'scope':p['scope'],'protocol_sha256':sha(OUT/'protocol.json')};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
