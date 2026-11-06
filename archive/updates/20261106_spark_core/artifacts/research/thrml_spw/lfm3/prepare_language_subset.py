"""Freeze a modest language subset before LFM3 stochastic benchmark scoring."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'language_subset';out.mkdir(exist_ok=True)
    source=Path('data/wikitext/test.parquet')
    protocol={'source_sha256':sha(Path(__file__)),'dataset_path':str(source),'dataset_sha256':sha(source),
        'checkpoint_revision':'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'selection_seed':202610295,'windows':8,'predicted_tokens_per_window':64,
        'selection':'Uniform sample without replacement from nonoverlapping65-token tiles of newline-joined nonempty text rows, no chat template or special tokens.',
        'purpose':'Exploratory small subset (512 predicted tokens), not standard full WikiText perplexity. Dataset previously consumed by other model experiments. No confirmatory equivalence test.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    tokenizer=AutoTokenizer.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',local_files_only=True,trust_remote_code=True)
    rows=pq.read_table(source,columns=['text'])['text'].to_pylist()
    text='\n'.join(s for s in rows if s and s.strip())
    ids=np.asarray(tokenizer(text,add_special_tokens=False)['input_ids'],dtype=np.int64)
    tiles=len(ids)//65;selected=np.sort(np.random.default_rng(202610295).choice(tiles,8,replace=False))
    tokens=np.stack([ids[k*65:(k+1)*65] for k in selected])
    np.savez_compressed(out/'tokens.npz',tokens=tokens,token_offsets=selected*65)
    (out/'manifest.json').write_text(json.dumps({'protocol_sha256':sha(out/'protocol.json'),
        'tokens_sha256':sha(out/'tokens.npz'),'token_count':len(ids),'selected_offsets':(selected*65).tolist(),
        'total_predicted_tokens':512,'status':'selection_frozen_no_LFM3_language_scores_yet'},indent=2)+'\n')
    print('Frozen8windows/512predictedtokens',flush=True)


if __name__=='__main__':main()
