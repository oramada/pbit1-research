"""Reserve another immutable confirmation set, excluding all consumed positions."""
import json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .transformer_export import MODEL_REVISION,DATASET_REVISION


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out=Path('data/thrml_spw/quarter_confirmation');out.mkdir(exist_ok=True)
    evidence=Path('results/thrml_spw/quarter_confirmation_preparation');evidence.mkdir(exist_ok=True)
    if (out/'tokens.npz').exists():raise RuntimeError('Reserved confirmation already exists; reuse it')
    path=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=DATASET_REVISION,local_files_only=True))
    prior_a=Path('data/thrml_spw/transformer_projection/export.npz');prior_b=Path('data/thrml_spw/transformer_confirmation/tokens.npz')
    a=np.load(prior_a);b=np.load(prior_b);prior=np.concatenate([a['test_token_offsets'],b['token_offsets']])
    original=json.loads(Path('results/thrml_spw/transformer_preparation/manifest.json').read_text());assert sha(path)==original['dataset_sha256']['test']
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f'models--distilbert--distilgpt2/snapshots/{MODEL_REVISION}',local_files_only=True)
    texts=pq.read_table(path,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(s for s in texts if s and s.strip()),add_special_tokens=False)['input_ids'],dtype=np.int64)
    legacy=np.sort(np.random.default_rng(17).choice(len(ids)-64,size=4,replace=False))
    for archive,offset_key,token_key in [(a,'test_token_offsets','test_tokens'),(b,'token_offsets','tokens')]:
        for offset,tokens in zip(archive[offset_key],archive[token_key]):np.testing.assert_array_equal(tokens,ids[offset:offset+65])
    eligible=np.setdiff1d(np.arange(len(ids)//65)*65,prior)
    eligible=eligible[np.array([not any(start<old+65 and old<start+65 for old in legacy) for start in eligible])]
    offsets=np.sort(np.random.default_rng(20261107).choice(eligible,128,replace=False));tokens=np.stack([ids[start:start+65] for start in offsets])
    assert not set(offsets)&set(prior);assert not any(start<old+65 and old<start+65 for start in offsets for old in legacy)
    np.savez_compressed(out/'tokens.npz',tokens=tokens,token_offsets=offsets)
    protocol={'model_revision':MODEL_REVISION,'dataset_revision':DATASET_REVISION,'parquet_sha256':sha(path),'tokens_sha256':sha(out/'tokens.npz'),
        'excluded_native_export_sha256':sha(prior_a),'excluded_first_confirmation_sha256':sha(prior_b),'prior_native_windows':32,'prior_first_confirmation_windows':128,'legacy_arbitrary_start_windows':legacy.tolist(),
        'selection_seed':20261107,'reserved_windows':128,'target_tokens':8192,'confirmation_scores_computed':False,
        'selection':'Uniform 65-token tiles after excluding all consumed native/confirmation tiles and positional overlaps with the documented four-window legacy pilot. No model inference on reserved tokens.',
        'evaluation_plan':'Freeze quarter-model development-selected T, eight-bit bound4 controls and group128. Evaluate FP32, affine4 block128, affine8 block128, programmed mean8 and native eight-bit selected-T with five seeds. No confirmation tuning; report every outcome.',
        'scope':'New token positions in the same WikiText-2 test corpus, not a new corpus or pretraining-clean guarantee. First three blocks, 25.92% parameters.',
        'source_sha256':sha(__file__)}
    (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:protocol[k] for k in ['reserved_windows','target_tokens','prior_native_windows','prior_first_confirmation_windows','confirmation_scores_computed','tokens_sha256']}))

if __name__=='__main__':main()
