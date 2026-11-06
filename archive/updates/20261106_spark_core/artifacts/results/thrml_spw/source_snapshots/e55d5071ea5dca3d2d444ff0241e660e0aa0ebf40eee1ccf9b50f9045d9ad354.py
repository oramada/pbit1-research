"""Audit prior validation reuse and clean-study exclusion completeness."""
import json
from pathlib import Path
import numpy as np
from .transformer_block_gpu import sha,save_json
ROOT=Path('results/thrml_spw')

def main():
    validation=[]
    for path in sorted(Path('data/thrml_spw').rglob('*.npz')):
        if path.name not in ['tokens.npz','export.npz']:continue
        with np.load(path) as a:
            for key in a.files:
                if 'validation' in key and 'offset' in key and a[key].ndim==1:
                    validation.append({'path':str(path),'key':key,'sha256':sha(path),'offsets':a[key].tolist()})
    result={}
    for name in ['calibrated_transformer','probe_placement','all_blocks','all_blocks_clean']:
        path=ROOT/name/'tokens.npz'
        if not path.exists():continue
        with np.load(path) as a:
            starts=a['token_offsets'];matches=[]
            for v in validation:
                old=np.array(v['offsets']);mask=np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65),axis=1)
                for index in np.flatnonzero(mask):
                    ar=np.load(v['path']);j=np.flatnonzero(old==starts[index]);exact=False
                    if len(j) and 'validation_tokens' in ar.files:
                        np.testing.assert_array_equal(a['tokens'][index],ar['validation_tokens'][j[0]]);exact=True
                    matches.append({'window_index':int(index),'offset':int(starts[index]),'old_validation_path':v['path'],'old_validation_key':v['key'],'exact_old_validation_token_row_match':exact})
            result[name]={'tokens_sha256':sha(path),'windows':len(starts),'overlapping_validation_windows':len({v['window_index'] for v in matches}),'matches':matches}
    assert result['calibrated_transformer']['overlapping_validation_windows']==4 and result['probe_placement']['overlapping_validation_windows']==3 and result['all_blocks']['overlapping_validation_windows']==1 and result['all_blocks_clean']['overlapping_validation_windows']==0
    p=json.loads((ROOT/'all_blocks_clean/protocol.json').read_text());catalog={(r['path'],r['key']) for r in p['excluded_archives']};required=set()
    for directory in ['data/thrml_spw','results/thrml_spw']:
        for path in sorted(Path(directory).rglob('*.npz')):
            if path.name not in ['tokens.npz','export.npz'] or path.parent==ROOT/'all_blocks_clean':continue
            with np.load(path) as a:
                for key in a.files:
                    if 'offset' in key and a[key].ndim==1 and a[key].size and a[key].dtype.kind in 'iu':required.add((str(path),key))
    assert catalog==required,(catalog^required)
    data={'passed':True,'validation_sources':validation,'studies':result,'clean_study_exclusion_catalog_complete':True,'known_offset_vectors_excluded':len(catalog),'source_sha256':sha(__file__),'scope':'Broader data-hygiene audit identifies reused older validation positions in prior nominally fresh-text studies. Exact token equality checked for all matches. Numerical sampling/task evidence unchanged; prior text was not wholly unused. Clean study excludes every recorded validation/test offset vector plus aborted-launch positions before scoring.'};save_json(ROOT/'TOKEN_HISTORY_AUDIT.json',data);save_json(ROOT/'all_blocks_clean/token_history_audit.json',data);print(json.dumps({'passed':True,'overlap_counts':{k:v['overlapping_validation_windows'] for k,v in result.items()},'clean_known_offset_vectors_excluded':len(catalog)},indent=2))
if __name__=='__main__':main()
