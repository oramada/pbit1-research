"""Domain-aware token audit: numeric offsets from different splits are not reuse."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub import hf_hub_download
from .transformer_block_gpu import sha,save_json
ROOT=Path('results/thrml_spw')

def main():
    p=json.loads((ROOT/'all_blocks_clean/protocol.json').read_text());tok=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True);streams={};sources={}
    for domain in ['validation','test']:
        path=Path(hf_hub_download('Salesforce/wikitext',f'wikitext-2-raw-v1/{domain}-00000-of-00001.parquet',repo_type='dataset',revision=p['dataset_revision'],local_files_only=True));text=pq.read_table(path,columns=['text']).column('text').to_pylist();streams[domain]=np.asarray(tok('\n'.join(t for t in text if t and t.strip()),add_special_tokens=False)['input_ids'],np.int64);sources[domain]={'path':str(path),'sha256':sha(path)}
    catalog={(r['path'],r['key']) for r in p['excluded_archives']};required=set();offsets=[]
    for directory in ['data/thrml_spw','results/thrml_spw']:
        for path in sorted(Path(directory).rglob('*.npz')):
            if path.name not in ['tokens.npz','export.npz'] or path.parent==ROOT/'all_blocks_clean':continue
            with np.load(path) as ar:
                for key in ar.files:
                    if 'offset' not in key or ar[key].ndim!=1 or not ar[key].size or ar[key].dtype.kind not in 'iu':continue
                    required.add((str(path),key));domain='validation' if 'validation' in key else 'test';tokenkey='validation_tokens' if domain=='validation' else ('test_tokens' if 'test_tokens' in ar.files else 'tokens');starts=ar[key];expected=np.stack([streams[domain][i:i+65] for i in starts]);np.testing.assert_array_equal(ar[tokenkey],expected)
                    offsets.append({'path':str(path),'key':key,'domain':domain,'windows':len(starts),'offsets':starts.tolist(),'source_sha256':sources[domain]['sha256'],'token_rows_verified_against_correct_split':True})
    assert required==catalog,(required^catalog)
    studies={}
    for name in ['calibrated_transformer','probe_placement','all_blocks','all_blocks_clean']:
        path=ROOT/name/'tokens.npz';ar=np.load(path);starts=ar['token_offsets'];np.testing.assert_array_equal(ar['tokens'],np.stack([streams['test'][i:i+65] for i in starts]));numeric=[]
        for r in offsets:
            if r['domain']!='validation':continue
            old=np.array(r['offsets']);mask=np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65),axis=1)
            for i in np.flatnonzero(mask):
                j=np.flatnonzero(old==starts[i]);same=False
                if len(j):same=bool(np.array_equal(ar['tokens'][i],streams['validation'][old[j[0]]:old[j[0]]+65]));assert not same
                numeric.append({'window_index':int(i),'test_offset':int(starts[i]),'validation_archive':r['path'],'matching_numeric_offset_is_same_text':same})
        studies[name]={'windows':len(starts),'numeric_cross_split_coincidences':len({v['window_index'] for v in numeric}),'cross_split_coincidences_are_not_identical_token_rows':True,'matches':numeric}
    clean=np.load(ROOT/'all_blocks_clean/tokens.npz')['token_offsets'];trueold=[]
    for r in offsets:
        if r['domain']=='test':trueold.extend(r['offsets'])
    old=np.array(trueold);assert not np.any((clean[:,None]<old[None]+65)&(old[None]<clean[:,None]+65))
    for name in ['calibrated_transformer','probe_placement']:
        ar=np.load(ROOT/name/'tokens.npz');cp=json.loads((ROOT/name/'protocol.json').read_text());old=np.concatenate([np.load(r['path'])[r['key']] for r in cp['excluded_archives']]+[np.array(cp['legacy_excluded_offsets'])]);x=ar['token_offsets'];assert not np.any((x[:,None]<old[None]+65)&(old[None]<x[:,None]+65))
    result={'passed':True,'correct_split_sources':sources,'all_archived_offset_token_rows_verified':True,'clean_test_windows_disjoint_from_every_recorded_test_position':True,'clean_exclusion_catalog_complete':True,'known_offset_vectors_excluded':len(catalog),'archived_offset_sources':offsets,'studies':studies,'source_sha256':sha(__file__),'scope':'Numeric offsets are relative to their dataset split. Earlier suspected validation reuse was a false alarm: matched offsets refer to different validation/test token rows. No contamination correction to previous numerical studies is warranted by this audit. The stopped launch and import failure remain preserved. New clean study conservatively excludes all numeric vectors from both domains plus stopped-launch test positions.'};save_json(ROOT/'TOKEN_HISTORY_AUDIT.json',result);save_json(ROOT/'all_blocks_clean/token_history_audit.json',result);print(json.dumps({'passed':True,'offset_vectors_checked':len(offsets),'cross_split_numeric_matches_are_false_alarm':True,'clean_test_overlap':0},indent=2))
if __name__=='__main__':main()
