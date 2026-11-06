"""Pin LFM3-native training windows and a complete article-bounded WikiText task."""
from pathlib import Path
import json,hashlib
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.broad.data_layout_v1 import articles_from_rows,causal_windows

def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    checkpoint=Path('data/thrml_spw/lfm3_1p7b/checkpoint');tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=True)
    root=OUT/'lfm3_data_v1';root.mkdir(exist_ok=False)
    train=Path('data/thrml_spw/lfm3_1p7b/adaptation_parquet/train.parquet');test=Path('data/wikitext/test.parquet')
    assert sha(test)=='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
    source={'source_sha256':sha(__file__),'parser_sha256':sha(Path('research/thrml_spw/broad/data_layout_v1.py')),'tokenizer_sha256':sha(checkpoint/'tokenizer.json'),'training_sha256':sha(train),'test_sha256':sha(test),
        'scope':'Preparationonly; LFM3tokenizer, noaddedspecialtokens, allwithinarticletargetsonce. Testwasusedinearlierresearch,notnewunexposeddata. No model scoring or chosen encoding.'}
    (root/'protocol.json').write_text(json.dumps(source,indent=2)+'\n')
    train_articles=articles_from_rows(pq.read_table(train,columns=['text'])['text'].to_pylist());order=np.random.default_rng(2026111701).permutation(len(train_articles));selected=[];short=[]
    for index in order:
        item=train_articles[int(index)];ids=tokenizer(item['text'],add_special_tokens=False)['input_ids']
        if len(ids)<257:continue
        short.append(ids[:257]);selected.append({'article_index':int(index),'text_sha256':hashlib.sha256(item['text'].encode()).hexdigest()})
        if len(short)==32:break
    assert len(short)==32
    with(root/'training_windows.npz').open('xb')as f:np.savez_compressed(f,input_ids=np.array(short,np.int64))
    articles=articles_from_rows(pq.read_table(test,columns=['text'])['text'].to_pylist());flat=[];offsets=[0];layout=[];windows=[];targets=0
    for index,item in enumerate(articles):
        ids=tokenizer(item['text'],add_special_tokens=False)['input_ids'];assert len(ids)>=2
        flat.extend(ids);offsets.append(len(flat));targets+=len(ids)-1
        layout.append({'article':index,'token_count':len(ids),'targets':len(ids)-1,'text_sha256':hashlib.sha256(item['text'].encode()).hexdigest()})
        windows.extend([{'article':index,**w}for w in causal_windows(len(ids),1024,512)])
    assert len(articles)==62
    with(root/'test_tokens.npz').open('xb')as f:np.savez_compressed(f,tokens=np.array(flat,np.int64),offsets=np.array(offsets,np.int64))
    (root/'test_layout.json').write_text(json.dumps({'articles':layout,'windows':windows,'targets':targets,'max_context':1024,'stride':512},indent=2)+'\n')
    result={'status':'LFM3_native_tokens_and_all62_article_task_prepared_without_scores','protocol_sha256':sha(root/'protocol.json'),'training_selection':selected,
        'training_windows_sha256':sha(root/'training_windows.npz'),'test_tokens_sha256':sha(root/'test_tokens.npz'),'test_layout_sha256':sha(root/'test_layout.json'),
        'test_articles':len(articles),'test_targets':targets,'test_windows':len(windows),'scope':source['scope']}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='training_selection'}))

if __name__=='__main__':main()
