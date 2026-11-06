"""Audit new conservative-study tokens and disjoint roles without scoring a model."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .verify_transformer_control import sha


def main():
    root=Path('results/thrml_spw');evidence=root/'quarter_conservative_preparation';p=json.loads((evidence/'protocol.json').read_text());data_path=Path('data/thrml_spw/quarter_conservative/tokens.npz');assert sha(data_path)==p['tokens_sha256'];data=np.load(data_path)
    assert sha(root/'quarter_confirmation/manifest.json')==p['motivating_failed_confirmation_sha256'];assert sha(root/'transformer_quarter/protocol.json')==p['source_development_protocol_sha256']
    lineage=[('data/thrml_spw/transformer_projection/export.npz',json.loads((root/'transformer_preparation/manifest.json').read_text())['export_sha256']),
        ('data/thrml_spw/transformer_confirmation/tokens.npz',json.loads((root/'transformer_confirmation/protocol.json').read_text())['input_export_sha256']),
        ('data/thrml_spw/quarter_confirmation/tokens.npz',json.loads((root/'quarter_confirmation/protocol.json').read_text())['input_export_sha256'])]
    for path,digest in lineage:assert sha(path)==digest
    old,first,quarter=[np.load(path) for path,_ in lineage]
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['model_revision']}",local_files_only=True)
    for split in ['validation','test']:
        source=Path(hf_hub_download('Salesforce/wikitext',f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet',repo_type='dataset',revision=p['dataset_revision'],local_files_only=True));assert sha(source)==p['source_parquet_sha256'][split]
        texts=pq.read_table(source,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(s for s in texts if s and s.strip()),add_special_tokens=False)['input_ids'])
        used=old['validation_token_offsets'] if split=='validation' else np.concatenate([old['test_token_offsets'],first['token_offsets'],quarter['token_offsets']])
        eligible=np.setdiff1d(np.arange(len(ids)//65)*65,used)
        if split=='test':
            legacy=np.sort(np.random.default_rng(17).choice(len(ids)-64,size=4,replace=False));assert legacy.tolist()==p['legacy_test_starts_excluded']
            eligible=eligible[~np.any((eligible[:,None]<legacy[None]+65)&(legacy[None]<eligible[:,None]+65),axis=1)]
        selected=np.sort(np.random.default_rng(p['selection_seeds'][split]).choice(eligible,128,replace=False));np.testing.assert_array_equal(selected,data[split+'_token_offsets']);assert data[split+'_tokens'].shape==(128,65)
        for start,tokens in zip(selected,data[split+'_tokens']):np.testing.assert_array_equal(tokens,ids[start:start+65]);assert not np.any((start<used+65)&(used<start+65))
    assert p['sample_search']==[1024,2048] and p['no_model_scores_computed']
    result={'passed':True,'validation_targets':8192,'test_targets':8192,'excluded_original_validation_windows':16,'excluded_consumed_test_windows':288,'excluded_legacy_windows':4,
        'excluded_archive_sha256':dict(lineage),'model_scores_computed':False,'source_sha256':sha(__file__),
        'scope':'All token/source/selection/offset hashes and positional exclusions verified; no model inference or new-study quality measured'}
    (evidence/'reservation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
