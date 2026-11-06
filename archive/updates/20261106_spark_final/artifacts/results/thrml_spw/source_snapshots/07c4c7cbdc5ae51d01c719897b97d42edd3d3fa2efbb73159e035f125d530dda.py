"""Reserve disjoint text tiles before inspecting confirmation model losses.

Only token IDs and offsets are prepared here. No model inference or test metrics
are computed; model/precision selection must finish on development data first.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .transformer_export import MODEL_REVISION,DATASET_REVISION


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out=Path('data/thrml_spw/transformer_confirmation');out.mkdir(exist_ok=True)
    evidence=Path('results/thrml_spw/transformer_confirmation_preparation');evidence.mkdir(exist_ok=True)
    if (out/'tokens.npz').exists():raise RuntimeError('Confirmation tokens already reserved; reuse immutable artifacts')
    prior=Path('data/thrml_spw/transformer_projection/export.npz');old=np.load(prior)
    source=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=DATASET_REVISION,local_files_only=True))
    original=json.loads(Path('results/thrml_spw/transformer_preparation/manifest.json').read_text())
    assert sha(source)==original['dataset_sha256']['test']
    snapshot=Path(HF_HUB_CACHE)/f'models--distilbert--distilgpt2/snapshots/{MODEL_REVISION}'
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    texts=pq.read_table(source,columns=['text']).column('text').to_pylist()
    ids=np.asarray(tokenizer('\n'.join(s for s in texts if s and s.strip()),add_special_tokens=False)['input_ids'],dtype=np.int64)
    width=65;prior_offsets=old['test_token_offsets'];assert np.all(prior_offsets%width==0)
    for tokens,offset in zip(old['test_tokens'],prior_offsets):np.testing.assert_array_equal(tokens,ids[offset:offset+width])
    eligible=np.setdiff1d(np.arange(len(ids)//width),prior_offsets//width)
    chosen=np.sort(np.random.default_rng(20261105).choice(eligible,128,replace=False));offsets=chosen*width
    assert not set(offsets)&set(prior_offsets)
    tokens=np.stack([ids[offset:offset+width] for offset in offsets]);np.savez_compressed(out/'tokens.npz',tokens=tokens,token_offsets=offsets)
    protocol={'model_revision':MODEL_REVISION,'dataset_revision':DATASET_REVISION,'source_test_parquet_sha256':sha(source),
        'prior_export_sha256':sha(prior),'tokens_sha256':sha(out/'tokens.npz'),'windows':128,'target_tokens':8192,'context_target_length':64,
        'selection_seed':20261105,'prior_test_windows_excluded':len(prior_offsets),'zero_token_overlap_with_prior_tiles':True,
        'selection':'Uniform tiles without replacement after excluding all earlier inspected test tiles. No logits, labels-as-performance, or losses inspected.',
        'evaluation_plan':'After finite-control calibration selection is frozen, compare FP32, conventional affine_zero_block128_4, floating controls T256, and selected finite controls T256 on these same windows. Five Monte Carlo seeds for native conditions; no further tuning on confirmation scores.',
        'scope':'Disjoint development/confirmation text tiles from the same WikiText-2 corpus; not a new dataset, pretraining-clean guarantee, or independent corpus-population sample.',
        'source_sha256':sha(__file__)}
    (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:protocol[k] for k in ('windows','target_tokens','prior_test_windows_excluded','zero_token_overlap_with_prior_tiles','tokens_sha256')}))

if __name__=='__main__':main()
