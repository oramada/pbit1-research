"""Audit reserved confirmation tiles without scoring any confirmation tokens."""
import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer,AutoModelForCausalLM
from .transformer_export import token_metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path('results/thrml_spw/transformer_confirmation_preparation');protocol=json.loads((root/'protocol.json').read_text())
    source=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=protocol['dataset_revision'],local_files_only=True))
    assert sha(source)==protocol['source_test_parquet_sha256']==sha('data/wikitext/test.parquet')
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{protocol['model_revision']}"
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    texts=pq.read_table(source,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(s for s in texts if s and s.strip()),add_special_tokens=False)['input_ids'])
    path=Path('data/thrml_spw/transformer_confirmation/tokens.npz');assert sha(path)==protocol['tokens_sha256']
    fresh=np.load(path);old=np.load('data/thrml_spw/transformer_projection/export.npz')
    offsets=fresh['token_offsets'];assert len(offsets)==len(np.unique(offsets))==128
    assert not set(offsets)&set(old['test_token_offsets'])
    for tokens,offset in zip(fresh['tokens'],offsets):np.testing.assert_array_equal(tokens,ids[offset:offset+65])
    eligible=np.setdiff1d(np.arange(len(ids)//65),old['test_token_offsets']//65)
    expected=np.sort(np.random.default_rng(protocol['selection_seed']).choice(eligible,128,replace=False))*65
    np.testing.assert_array_equal(offsets,expected)
    # The older generic pilot used four randomly started windows, not aligned tiles.
    # Reconstruct its documented command/default seed and reproduce its known score.
    legacy=np.sort(np.random.default_rng(17).choice(len(ids)-64,size=4,replace=False))
    assert not any(a<b+65 and b<a+65 for a in offsets for b in legacy)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();nll=[]
    with torch.inference_mode():
        for start in legacy:
            token=torch.from_numpy(ids[start:start+65][None].copy());nll.append(token_metrics(model(input_ids=token,use_cache=False).logits,token)[0])
    summary=dict(line.split(': ',1) for line in Path('results/transformer/run_summary.txt').read_text().splitlines())
    ppl=float(np.exp(np.mean(np.stack(nll).astype(np.float64))));np.testing.assert_allclose(ppl,float(summary['fp_perplexity']),rtol=1e-5)
    result={'passed':True,'reserved_windows':128,'target_tokens':8192,'native_prior_windows_excluded':32,'legacy_prior_windows_checked':4,
        'legacy_starts':legacy.tolist(),'legacy_baseline_reproduced_perplexity':ppl,'confirmation_model_scores_computed':False,
        'scope':'Exact token/offset/hash/selection checks and zero positional overlap with native test tiles and the documented older generic pilot; only old pilot windows were scored.',
        'source_sha256':sha(__file__)}
    (root/'reservation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
