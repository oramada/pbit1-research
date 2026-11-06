"""Independently check domain-specific archives, exclusions and fixed budget."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/sample_allocation')

def main():
    p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'tokens.npz')==p['tokens_sha256'];ar=np.load(OUT/'tokens.npz');tok=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True)
    observed=set()
    for directory in ['data/thrml_spw','results/thrml_spw']:
        for path in Path(directory).rglob('*.npz'):
            if path.name not in ['tokens.npz','export.npz'] or path.parent==OUT:continue
            with np.load(path) as a:
                for key in a.files:
                    if 'offset' in key and a[key].ndim==1 and a[key].size and a[key].dtype.kind in 'iu':observed.add((str(path),key,'validation' if 'validation' in key else 'test'))
    assert observed=={(r['path'],r['key'],r['domain']) for r in p['token_history']}
    for domain in ['validation','test']:
        source=p['source_parquets'][domain];assert sha(source['path'])==source['sha256'];texts=pq.read_table(source['path'],columns=['text']).column('text').to_pylist();ids=np.asarray(tok('\n'.join(t for t in texts if t and t.strip()),add_special_tokens=False)['input_ids'],np.int64);starts=ar[domain+'_token_offsets'];assert len(starts)==32 and np.min(np.diff(starts))>=65;np.testing.assert_array_equal(ar[domain+'_tokens'],np.stack([ids[i:i+65] for i in starts]))
        for r in p['token_history']:
            if r['domain']!=domain:continue
            assert sha(r['path'])==r['sha256'];a=np.load(r['path']);old=a[r['key']];assert not np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65));key='validation_tokens' if domain=='validation' else ('test_tokens' if 'test_tokens' in a.files else 'tokens');np.testing.assert_array_equal(a[key],np.stack([ids[i:i+65] for i in old]))
        if domain=='test':
            old=np.array(p['legacy_excluded_test_offsets']);assert not np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65))
    old=json.loads(Path('results/thrml_spw/all_blocks_clean/protocol.json').read_text());assert p['control_artifacts']==old['control_artifacts'];assert json.loads(Path('results/thrml_spw/all_blocks_clean/preparation_audit.json').read_text())['passed']
    for c in p['control_artifacts']:assert sha(c['path'])==c['sha256']
    units=[c['weights']//589824 for c in p['control_artifacts']];assert sum(units)*4==288 and all(c['weights']%589824==0 for c in p['control_artifacts']);assert sum(c['weights'] for c in p['control_artifacts'])==42467328
    for name,h in p['source_sha256'].items():assert sha(name)==h
    r={'passed':True,'new_validation_windows':32,'new_test_windows':32,'all_old_token_rows_verified_in_correct_split':True,'all_prior_offset_vectors_catalogued':True,'controls_unchanged_and_previously_weight_verified':True,'exact_budget_per_bank':43486543872,'source_sha256':sha(__file__),'no_model_scores_or_native_samples':True};save_json(OUT/'preparation_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
