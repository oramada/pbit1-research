"""Independently verify LFM3 token identity, article separation and full coverage."""
from pathlib import Path
import json
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.broad.data_layout_v1 import articles_from_rows

def main():
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    root=OUT/'lfm3_data_v1';result=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text());layout=json.loads((root/'test_layout.json').read_text())
    for file,key in [('test_tokens.npz','test_tokens_sha256'),('training_windows.npz','training_windows_sha256'),('test_layout.json','test_layout_sha256')]:assert sha(root/file)==result[key]
    tokenizer=AutoTokenizer.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',local_files_only=True,trust_remote_code=True)
    import hashlib
    test=articles_from_rows(pq.read_table('data/wikitext/test.parquet',columns=['text'])['text'].to_pylist())
    with np.load(root/'test_tokens.npz')as f:tokens=f['tokens'];offsets=f['offsets']
    covered=0
    for i,article in enumerate(test):
        ids=tokenizer(article['text'],add_special_tokens=False)['input_ids'];a,b=map(int,offsets[i:i+2]);assert np.array_equal(tokens[a:b],ids)
        assert hashlib.sha256(article['text'].encode()).hexdigest()==layout['articles'][i]['text_sha256']
        counts=np.zeros(len(ids)-1,np.int8)
        for w in [w for w in layout['windows']if w['article']==i]:
            assert 0<=w['context_start']<w['target_start']<w['target_stop']<=w['context_stop']<=len(ids)
            assert w['context_stop']-w['context_start']<=1024
            counts[w['target_start']-1:w['target_stop']-1]+=1
        assert np.all(counts==1);covered+=len(counts)
    assert covered==294442==layout['targets']and len(layout['windows'])==607
    train=articles_from_rows(pq.read_table('data/thrml_spw/lfm3_1p7b/adaptation_parquet/train.parquet',columns=['text'])['text'].to_pylist())
    with np.load(root/'training_windows.npz')as f:windows=f['input_ids']
    assert windows.shape==(32,257)
    test_hashes={x['text_sha256']for x in layout['articles']}
    for row,chosen in enumerate(result['training_selection']):
        article=train[chosen['article_index']];assert chosen['text_sha256']not in test_hashes
        assert hashlib.sha256(article['text'].encode()).hexdigest()==chosen['text_sha256']
        assert np.array_equal(windows[row],tokenizer(article['text'],add_special_tokens=False)['input_ids'][:257])
    out={'status':'all_tokens_retokenized_and_294442_targets_covered_once','source_sha256':sha(__file__),'preparation_sha256':sha(root/'results.json'),'test_articles':62,'test_windows':607,'test_targets':covered,'training_windows':32,'scope':'Dataauditonly; no LFM3neuralqualityresult.'}
    (OUT/'lfm3_data_audit_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
