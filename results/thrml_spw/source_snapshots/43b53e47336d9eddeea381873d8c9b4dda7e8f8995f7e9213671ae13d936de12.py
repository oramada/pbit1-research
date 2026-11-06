"""Complete public-state causal regeneration for compact word-case receipts."""
import json,os,fcntl
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.word_comparison.common import OUT,PARENT,TRAIN,sha,write,guard,parent_closed,keys
from results.thrml_spw.fashion_cnn_native.word_reference.native import encode
from results.thrml_spw.fashion_cnn_native.word_reference.witness import recover
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and m['protocol_sha256']==sha(OUT/'protocol.json');assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0'
    with (OUT/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ap=OUT/'audit_progress.json';a=json.loads(ap.read_text()) if ap.exists() else {'rows':[],'public_updates':0,'counter_entries_exact':0,'protocol_sha256':sha(OUT/'protocol.json')};assert a['protocol_sha256']==sha(OUT/'protocol.json');done={r['name']:r for r in a['rows']};train=json.loads((TRAIN/'manifest.json').read_text());x=np.load(PARENT/'test_data.npz')['inputs'].astype(float).reshape(1250,8,1,14,14)
        for row in m['rows']:
            assert sha(row['path'])==row['sha256']
            if row['name'] in done:assert done[row['name']]['case_sha256']==row['sha256'];continue
            guard(p);model=np.load(train['rows'][row['model']]['path']);enc=encode(model);saved=np.load(row['path']);hashes,logits=saved['receipts'],saved['logits'];clip=np.zeros(3,np.int64);entries=0
            for first in range(0,1250,4):
                n=min(4,1250-first);kk=[keys(row['model'],row['seed'],row['T'],li,first,n) for li in range(3)];raw=recover(x[first:first+n],model,row['T'],kk,enc);np.testing.assert_array_equal(receipt(raw),hashes[first:first+n]);np.testing.assert_array_equal(raw['output_2'].reshape(n*8,10),logits[first*8:(first+n)*8]);entries+=sum(raw[f'gate_{li}'].size+raw[f'column_{li}'].size for li in range(3))
                for li,h in enumerate([x[first:first+n],raw['output_0'],raw['output_1']]):clip[li]+=np.count_nonzero((h<0)|(h>float(model['activation_span'][li])))
            assert entries==47182500;np.testing.assert_array_equal(clip,saved['clipped_coordinates']);np.testing.assert_array_equal(clip,row['clipped_coordinates']);a['counter_entries_exact']+=entries;a['public_updates']+=row['native_updates'];a['rows'].append({'name':row['name'],'case_sha256':row['sha256'],'all17500_bank_array_digests_exact':True,'counter_entries_exact':entries,'public_updates':row['native_updates']});write(ap,a);print(json.dumps({'audited_cases':len(a['rows']),'last':row['name']}),flush=True)
        assert len(a['rows'])==150 and a['public_updates']==223776000000 and a['counter_entries_exact']==7077375000;guard(p);write(OUT/'native_audit.json',{'passed':True,**a,'manifest_sha256':sha(OUT/'manifest.json'),'scope':'All compact native counter/code/mask/canonical-output arrays fully regenerated with separate public raw states. Hashes do not decode arrays; deterministic regeneration required. All15million final logits checked directly. No independent extra quality replicas or hardware measurement.'})
if __name__=='__main__':main()
