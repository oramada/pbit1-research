"""Verify compact receipts by reconstructing all original causal arrays."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt,NAMES
from results.thrml_spw.fashion_cnn_native.compact.recover import recover
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'compact_witness'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_hash(name,a):
    header='{"dtype":'+json.dumps(a.dtype.str)+',"name":'+json.dumps(name)+',"shape":'+json.dumps(list(a.shape),separators=(',',':'))+'}'
    return np.frombuffer(hashlib.sha256(header.encode()+b'\0'+np.asarray(a,order='C').tobytes()).digest(),np.uint8)
def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'No duplicate witness'
    audit=json.loads((ROOT/'preflight/evidence_audit.json').read_text());assert audit['passed'];parent=json.loads((ROOT/'preflight/protocol.json').read_text());training=json.loads((ROOT/'training/manifest.json').read_text())
    sources={str(f):sha(f) for f in (ROOT/'compact').glob('*.py')};sources.update(parent['source_sha256'])
    p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'parent_audit_sha256':sha(ROOT/'preflight/evidence_audit.json'),'parent_protocol_sha256':sha(ROOT/'preflight/protocol.json'),'training_manifest_sha256':sha(ROOT/'training/manifest.json'),'array_order':list(NAMES),'format':'uint8[bank,14,32] SHA256 digests of sorted compact JSON(name,dtype,shape),NUL,C-contiguous bytes. Full patch arrays are NOT decodable from hashes; regenerate from pinned public sampler+keys+inputs+model and accept only exact digest matches.','scope':'All15 original validation witnesses fully reconstructed, no new native observer execution and no test labels/task scoring. Extra public replay is charged separately; this is recovery/receipt validation, not independent quality replications.'}
    (OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');x=np.load(ROOT/'training/validation.npz')['inputs'][:8].astype(float).reshape(2,4,1,14,14);rows=[];updates=0;array_count=0;coordinate_count=0
    for row in audit['rows']:
        mi,t=row['model'],row['T'];assert sha(row['path'])==row['sha256'];original=dict(np.load(row['path']));fingerprints=receipt(original)
        # Different serializer construction and hashing calls validate the format.
        for ch in range(2):
            for ki,name in enumerate(NAMES):np.testing.assert_array_equal(fingerprints[ch,ki],independent_hash(name,original[name][ch]))
        key=jax.random.fold_in(jax.random.fold_in(jax.random.key(parent['key_root']),mi),t);keys=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)];modelrow=training['rows'][mi];assert sha(modelrow['path'])==modelrow['sha256'];model=np.load(modelrow['path']);reconstructed=recover(x,model,t,keys)
        np.testing.assert_array_equal(receipt(reconstructed),fingerprints)
        for name in NAMES:np.testing.assert_array_equal(original[name],reconstructed[name]);coordinate_count+=original[name].size
        # Detect value, shape, dtype and semantic-name changes, even with reused bytes.
        a=original['gate_0'][0];before=independent_hash('gate_0',a);changed=a.copy();changed.flat[0]+=1
        assert not np.array_equal(before,independent_hash('gate_0',changed));assert not np.array_equal(before,independent_hash('gate_0',a.ravel()));assert not np.array_equal(before,independent_hash('gate_0',a.view(np.uint32)));assert not np.array_equal(before,independent_hash('gate_1',a))
        path=OUT/f'model{mi}_T{t}.npz';np.savez_compressed(path,receipts=fingerprints,logits=original['output_2']);u=2*t*2664;updates+=u;array_count+=2*len(NAMES);rows.append({'model':mi,'T':t,'path':str(path),'sha256':sha(path),'arrays_exact':2*len(NAMES),'new_public_updates':u});print(json.dumps(rows[-1]),flush=True)
    for f,h in sources.items():assert sha(f)==h
    assert len(rows)==15 and array_count==420 and updates==1918080
    result={'passed':True,'rows':rows,'arrays_recovered_byte_exact':array_count,'coordinates_recovered_exact':coordinate_count,'extra_public_replay_updates':updates,'new_native_observer_updates':0,'format_mutation_checks':60,'full_cohort_receipt_bytes_uncompressed':1250*14*32*150,'full_cohort_logits_bytes_uncompressed':10000*10*8*150,'scope':p['scope'],'protocol_sha256':sha(OUT/'protocol.json')}
    (OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
