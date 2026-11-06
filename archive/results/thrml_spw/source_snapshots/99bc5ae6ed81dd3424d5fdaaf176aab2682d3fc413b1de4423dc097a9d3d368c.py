"""Independent source-text and exclusion audit; performs no model scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE
OUT=Path('results/thrml_spw/fine_range_confirmation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'tokens.npz')==p['tokens_sha256'];source=p['source_parquet'];assert sha(source['path'])==source['sha256'];a=np.load(OUT/'tokens.npz');starts=a['test_token_offsets'];tokens=a['test_tokens'];assert tokens.shape==(32,65) and starts.shape==(32,) and len(set(starts.tolist()))==32 and np.all(starts%65==0)
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True);lines=pq.read_table(source['path'],columns=['text']).column('text').to_pylist();ids=np.array(tokenizer('\n'.join(s for s in lines if s and s.strip()),add_special_tokens=False)['input_ids'],dtype=np.int64);np.testing.assert_array_equal(tokens,np.array([ids[int(i):int(i)+65] for i in starts]))
    prior=[];files=0
    for item in p['token_history']:
        assert sha(item['path'])==item['sha256'];old=np.load(item['path'])[item['key']];assert len(old)==item['windows'];files+=1
        if item['domain']=='test':
            assert not np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65));prior.extend(old.tolist())
    legacy=np.array(p['legacy_excluded_test_offsets']);assert not np.any((starts[:,None]<legacy[None]+65)&(legacy[None]<starts[:,None]+65));prior.extend(legacy.tolist());prior=np.array(prior);eligible=np.arange(len(ids)//65)*65;eligible=eligible[~np.any((eligible[:,None]<prior[None]+65)&(prior[None]<eligible[:,None]+65),axis=1)];chosen=np.sort(np.random.default_rng(2026118001).choice(eligible,32,replace=False));np.testing.assert_array_equal(chosen,starts)
    assert p['conditions']==[{'name':'coarse','group':128,'samples':1024},{'name':'fine','group':16,'samples':512}] and p['sampling_seeds']==list(range(5)) and p['converted_weights']==42467328
    for group,controls in p['control_artifacts'].items():
        assert len(controls)==24
        for c in controls:assert sha(c['path'])==c['sha256']
    r={'passed':True,'test_windows':32,'targets':2048,'all_tokens_reconstructed_from_pinned_test_stream':True,'domain_qualified_overlap_checks_passed':True,'prior_offset_records_checked':files,'eligible_unused_windows':len(eligible),'frozen_random_reservation_reproduced':True,'protocol_sha256':sha(OUT/'protocol.json'),'tokens_sha256':sha(OUT/'tokens.npz'),'source_sha256':sha(__file__),'model_scores_computed':False,'native_samples_collected':False,'scope':'Reservation/preparation only; no neural result or completed execution-source freeze.'};(OUT/'reservation_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
