"""Fetch pinned assets and freeze evaluation geometry; never load a neural model."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data_layout_v1 import articles_from_rows,causal_windows

BASE=Path('results/thrml_spw/broad_verification_20261105')
OUT=BASE/'additional_assets_v1'
DATA=Path('data/thrml_spw/broad_additional_v1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    inventory=BASE/'additional_model_inventory/pinned_hub_inventory.json'
    records=json.loads(inventory.read_text())['records']
    OUT.mkdir(exist_ok=False);DATA.mkdir(parents=True,exist_ok=True)
    locations={};files={}
    for record in records:
        directory=DATA/record['id'].replace('/','--')
        for item in record['files']:
            p=Path(hf_hub_download(record['id'],item['path'],revision=record['revision'],
                repo_type='dataset' if record['kind']=='dataset' else 'model',local_dir=directory))
            assert p.stat().st_size==item['bytes'],str(p)
            digest=sha(p)
            if item['lfs_sha256'] is not None:assert digest==item['lfs_sha256'],str(p)
            else:
                blob=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()
                assert blob==item['git_blob_id'],str(p)
            files[str(p)]={'sha256':digest,'bytes':p.stat().st_size,'revision':record['revision']}
        locations[record['id']]=directory
    model='openai-community/gpt2';tokenizer=AutoTokenizer.from_pretrained(locations[model],local_files_only=True)
    table=pq.read_table(locations['Salesforce/wikitext']/'wikitext-2-raw-v1/test-00000-of-00001.parquet',columns=['text'])
    articles=articles_from_rows(table['text'].to_pylist())
    tokens=[];offsets=[0];plans=[];article_records=[]
    for index,article in enumerate(articles):
        ids=tokenizer(article['text'],add_special_tokens=False)['input_ids']
        windows=causal_windows(len(ids))
        assert [t for w in windows for t in range(w['target_start'],w['target_stop'])]==list(range(1,len(ids)))
        plans.extend({'article':index,**w} for w in windows)
        tokens.extend(ids);offsets.append(len(tokens))
        article_records.append({'index':index,'source_row_start':article['source_row_start'],
            'text_sha256':hashlib.sha256(article['text'].encode()).hexdigest(),
            'token_count':len(ids),'target_count':max(0,len(ids)-1),'windows':len(windows)})
    gp=OUT/'gpt2_tokens.npz'
    with gp.open('xb') as f:np.savez_compressed(f,tokens=np.asarray(tokens,dtype=np.int64),offsets=np.asarray(offsets,dtype=np.int64))
    write(OUT/'gpt2_layout.json',{'model':model,'articles':article_records,'windows':plans,
        'targets':sum(v['target_count'] for v in article_records),'tokens_sha256':sha(gp),
        'policy':'Complete within-article causal targets1..N-1 once; max1024context/512stride; no cross-article predictions or added special tokens.'})
    model='distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer=AutoTokenizer.from_pretrained(locations[model],local_files_only=True)
    table=pq.read_table(locations['stanfordnlp/imdb']/'plain_text/test-00000-of-00001.parquet',columns=['text','label'])
    texts=table['text'].to_pylist();labels=np.asarray(table['label'].to_pylist(),dtype=np.int64)
    assert len(texts)==len(labels)==25000 and set(labels)=={0,1}
    ids=np.empty((25000,256),dtype=np.int64);mask=np.empty_like(ids);lengths=np.empty(25000,dtype=np.int64)
    assert tokenizer.padding_side=='right'
    for begin in range(0,25000,128):
        batch=texts[begin:begin+128];end=begin+len(batch)
        full=tokenizer(batch,add_special_tokens=True,padding=False,truncation=False)
        lengths[begin:end]=[len(v) for v in full['input_ids']]
        encoded=tokenizer(batch,add_special_tokens=True,padding='max_length',truncation=True,max_length=256,return_tensors='np')
        ids[begin:end]=encoded['input_ids'];mask[begin:end]=encoded['attention_mask']
    assert np.array_equal(mask.sum(1),np.minimum(lengths,256))
    assert np.all(np.diff(mask,axis=1)<=0) and np.all(ids[mask==0]==tokenizer.pad_token_id)
    ip=OUT/'distilbert_tokens.npz'
    with ip.open('xb') as f:np.savez_compressed(f,input_ids=ids,attention_mask=mask,labels=labels,untruncated_lengths=lengths)
    write(OUT/'distilbert_layout.json',{'model':model,'reviews':25000,'max_length':256,'padding_side':'right',
        'truncated_reviews':int((lengths>256).sum()),'truncation_fraction':float((lengths>256).mean()),
        'class_counts':np.bincount(labels,minlength=2).tolist(),'tokens_sha256':sha(ip),
        'source_text_sha256_by_row':[hashlib.sha256(t.encode()).hexdigest() for t in texts],
        'policy':'All25000labelledtestreviews in source row order, no filtering; truncation256with rightpadding and attention masks.'})
    for f in OUT.glob('*'):
        if f.is_file():files[str(f)]={'sha256':sha(f),'bytes':f.stat().st_size}
    sources=[Path(__file__),Path(__file__).with_name('data_layout_v1.py')]
    result={'status':'all_pinned_model_dataset_hashes_and_full_task_geometry_frozen_without_quality_scoring',
        'inventory_sha256':sha(inventory),'files':files,'sources_sha256':{str(p):sha(p) for p in sources},
        'checkpoint_locations':{k:str(v) for k,v in locations.items()},
        'gpt2_articles':len(articles),'gpt2_targets':sum(v['target_count'] for v in article_records),
        'distilbert_reviews':25000,'distilbert_truncation_fraction':float((lengths>256).mean()),
        'scope':'Pinned assets and deterministic full-test token/label geometry only. No model forward, adaptation, stochastic weights, quality outcome, R2operator or hardware evidence.'}
    write(OUT/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['files','checkpoint_locations']}))

if __name__=='__main__':main()
