"""Verify complete old benchmark geometry without any new model forwards."""
from pathlib import Path
import json,hashlib
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from research.thrml_spw.broad.data_layout_v1 import articles_from_rows,causal_windows

def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    old=Path('results/thrml_spw/broad_verification_20261105/additional_assets_v1_mirror/results/thrml_spw/broad_verification_20261105/additional_assets_v1')
    layout_file=old/'gpt2_layout.json';token_file=old/'gpt2_tokens.npz';layout=json.loads(layout_file.read_text());assert sha(token_file)==layout['tokens_sha256']
    raw=Path('data/wikitext/test.parquet');assert sha(raw)=='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
    tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True);articles=articles_from_rows(pq.read_table(raw,columns=['text'])['text'].to_pylist());assert len(articles)==len(layout['articles'])==62
    with np.load(token_file,allow_pickle=False)as d:tokens=d['tokens'].copy();offsets=d['offsets'].copy()
    windows=[];targets=0
    for index,(article,record)in enumerate(zip(articles,layout['articles'])):
        assert hashlib.sha256(article['text'].encode()).hexdigest()==record['text_sha256']
        ids=np.asarray(tokenizer(article['text'],add_special_tokens=False)['input_ids'],np.int64)
        assert np.array_equal(tokens[offsets[index]:offsets[index]+len(ids)],ids)
        assert record['token_count']==len(ids);targets+=len(ids)-1
        windows.extend([{'article':index,**w}for w in causal_windows(len(ids),1024,512)])
    assert windows==layout['windows']and targets==layout['targets']==283225
    output=OUT/'benchmark_preparation_v1.json'
    with output.open('x')as f:json.dump({'status':'verified_complete_62_article_283225_target_benchmark_without_new_forwards','source_sha256':sha(__file__),
        'raw_file':str(raw),'raw_sha256':sha(raw),'tokens':str(token_file),'tokens_sha256':sha(token_file),'layout':str(layout_file),'layout_sha256':sha(layout_file),'windows':len(windows),'targets':targets,
        'scope':'Benchmark was inspected in priorstudy. Prospective re-evaluation of a training/calibration-selectedfix, not newunexposedtestdata.'},f,indent=2)
    print(json.dumps({'prepared_targets':targets,'windows':len(windows)}))

if __name__=='__main__':main()
