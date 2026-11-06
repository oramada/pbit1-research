"""Independently verify unused quarter-model confirmation positions and provenance."""
import json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path('results/thrml_spw/quarter_confirmation_preparation');p=json.loads((root/'protocol.json').read_text())
    path=Path('data/thrml_spw/quarter_confirmation/tokens.npz');assert sha(path)==p['tokens_sha256'];new=np.load(path)
    a_path=Path('data/thrml_spw/transformer_projection/export.npz');b_path=Path('data/thrml_spw/transformer_confirmation/tokens.npz')
    assert sha(a_path)==p['excluded_native_export_sha256'] and sha(b_path)==p['excluded_first_confirmation_sha256']
    a,b=np.load(a_path),np.load(b_path);used=np.concatenate([a['test_token_offsets'],b['token_offsets']])
    source=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=p['dataset_revision'],local_files_only=True));assert sha(source)==p['parquet_sha256']
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['model_revision']}",local_files_only=True)
    texts=pq.read_table(source,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(x for x in texts if x and x.strip()),add_special_tokens=False)['input_ids'])
    old=np.sort(np.random.default_rng(17).choice(len(ids)-64,size=4,replace=False));assert old.tolist()==p['legacy_arbitrary_start_windows']
    eligible=[start for start in np.arange(len(ids)//65)*65 if start not in set(used) and not np.any((start<old+65)&(old<start+65))]
    expected=np.sort(np.random.default_rng(20261107).choice(np.array(eligible),128,replace=False));np.testing.assert_array_equal(new['token_offsets'],expected)
    assert new['tokens'].shape==(128,65)
    for offset,tokens in zip(new['token_offsets'],new['tokens']):
        np.testing.assert_array_equal(tokens,ids[offset:offset+65]);assert not np.any((offset<used+65)&(used<offset+65));assert not np.any((offset<old+65)&(old<offset+65))
    result={'passed':True,'reserved_windows':128,'target_tokens':8192,'excluded_native_windows':32,'excluded_consumed_confirmation_windows':128,'excluded_legacy_windows':4,
        'confirmation_model_scores_computed':False,'tokens_sha256':sha(path),'source_sha256':sha(__file__),
        'scope':'Exact selection, token, offset and source hashes; zero positional overlap with all consumed text sets. No model inference or quality scores computed.'}
    (root/'reservation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
