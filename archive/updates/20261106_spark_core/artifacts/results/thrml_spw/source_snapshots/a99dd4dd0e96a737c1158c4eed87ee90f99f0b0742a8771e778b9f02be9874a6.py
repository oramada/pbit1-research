"""Independent full-token/label rebuild against closed pinned source files."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

BASE=Path('results/thrml_spw/broad_verification_20261105')
MIRROR=BASE/'additional_assets_v1_mirror'
ASSETS=MIRROR/BASE/'additional_assets_v1'
DATA=MIRROR/'data/thrml_spw/broad_additional_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    download=json.loads((MIRROR/'download_manifest.json').read_text())
    assert download['source_manifest_sha256']==sha(BASE/'additional_assets_v1_manifest.json')
    for name,r in download['files'].items():
        p=MIRROR/name;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],name
    # Independent article segmentation by a primary-heading regular expression.
    rows=pq.read_table(DATA/'Salesforce--wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet')['text'].to_pylist()
    groups=[];current=[];first=None
    for index,text in enumerate(rows):
        if not text.strip():continue
        match=re.fullmatch(r'\s*=\s+(.+?)\s+=\s*',text)
        primary=(match is not None and match[1].split()[0]!='=' and match[1].split()[-1]!='=')
        if primary and current:groups.append((first,'\n'.join(current)));current=[]
        if not current:first=index
        current.append(text)
    if current:groups.append((first,'\n'.join(current)))
    layout=json.loads((ASSETS/'gpt2_layout.json').read_text())
    assert layout['tokens_sha256']==sha(ASSETS/'gpt2_tokens.npz')
    with np.load(ASSETS/'gpt2_tokens.npz',allow_pickle=False) as a:
        tokens,offsets=a['tokens'],a['offsets']
    assert tokens.dtype==offsets.dtype==np.int64 and offsets.shape==(len(groups)+1,)
    assert offsets[0]==0 and offsets[-1]==len(tokens) and np.all(np.diff(offsets)>=0)
    tokenizer=AutoTokenizer.from_pretrained(DATA/'openai-community--gpt2',local_files_only=True)
    expected_plan=[];total=0
    assert len(layout['articles'])==len(groups)==62
    for i,(row,text) in enumerate(groups):
        expected=np.asarray(tokenizer.encode(text,add_special_tokens=False),dtype=np.int64)
        np.testing.assert_array_equal(tokens[offsets[i]:offsets[i+1]],expected)
        item=layout['articles'][i];n=len(expected);windows=(max(0,n-1)+511)//512
        assert item=={'index':i,'source_row_start':row,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'token_count':n,'target_count':max(0,n-1),'windows':windows}
        coverage=np.zeros(n,dtype=np.int8)
        for j in range(windows):
            start=1+j*512;end=min(start+512,n);begin=max(0,end-1024)
            expected_plan.append({'article':i,'context_start':begin,'context_stop':end,'target_start':start,'target_stop':end})
            assert 0<=begin<start<end<=n and end-begin<=1024
            coverage[start:end]+=1
        assert not n or coverage[0]==0
        assert np.all(coverage[1:]==1)
        total+=max(0,n-1)
    assert layout['windows']==expected_plan and layout['targets']==total==283225
    layout=json.loads((ASSETS/'distilbert_layout.json').read_text())
    assert layout['tokens_sha256']==sha(ASSETS/'distilbert_tokens.npz')
    table=pq.read_table(DATA/'stanfordnlp--imdb/plain_text/test-00000-of-00001.parquet')
    texts=table['text'].to_pylist();labels=np.asarray(table['label'].to_pylist(),dtype=np.int64)
    assert len(texts)==25000
    assert layout['source_text_sha256_by_row']==[hashlib.sha256(t.encode()).hexdigest() for t in texts]
    with np.load(ASSETS/'distilbert_tokens.npz',allow_pickle=False) as a:
        ids,mask,actual_labels,lengths=[a[k] for k in ['input_ids','attention_mask','labels','untruncated_lengths']]
    np.testing.assert_array_equal(labels,actual_labels)
    assert ids.shape==mask.shape==(25000,256) and lengths.shape==(25000,)
    assert all(v.dtype==np.int64 for v in [ids,mask,actual_labels,lengths])
    tokenizer=AutoTokenizer.from_pretrained(DATA/'distilbert--distilbert-base-uncased-finetuned-sst-2-english',local_files_only=True)
    for begin in range(0,25000,97):
        batch=texts[begin:begin+97];end=begin+len(batch)
        # Different batching from producer128; reconstruct all tokens/lengths.
        encoded=tokenizer(batch,padding='max_length',truncation=True,max_length=256,return_tensors='np')
        np.testing.assert_array_equal(ids[begin:end],encoded['input_ids'])
        np.testing.assert_array_equal(mask[begin:end],encoded['attention_mask'])
        expected_lengths=[len(tokenizer.encode(t,add_special_tokens=True,truncation=False)) for t in batch]
        np.testing.assert_array_equal(lengths[begin:end],expected_lengths)
    assert set(np.unique(mask))=={0,1} and np.all(np.diff(mask,axis=1)<=0)
    assert np.all(ids[mask==0]==tokenizer.pad_token_id)
    np.testing.assert_array_equal(mask.sum(1),np.minimum(lengths,256))
    assert layout['reviews']==25000 and layout['max_length']==256 and layout['padding_side']=='right'
    assert layout['truncated_reviews']==int((lengths>256).sum())==10840
    assert layout['truncation_fraction']==float((lengths>256).mean())==.4336
    assert layout['class_counts']==np.bincount(labels,minlength=2).tolist()==[12500,12500]
    result={'status':'all_real_test_tokens_labels_article_targets_and_truncation_independently_rebuilt',
        'source_sha256':sha(Path(__file__)),'manifest_sha256':sha(BASE/'additional_assets_v1_manifest.json'),
        'download_manifest_sha256':sha(MIRROR/'download_manifest.json'),
        'hash_checked_nonweight_files':len(download['files']),'gpt2_articles':62,'gpt2_targets':total,
        'gpt2_all_targets_once_and_article_bounded':True,'distilbert_reviews':25000,'distilbert_all_token_rows_rebuilt':True,
        'distilbert_truncated_reviews':10840,'distilbert_class_counts':[12500,12500],
        'scope':'Independent local reconstruction of all test tokens/labels from pinned tokenizer/dataset bytes. Model weights were hash-checked by remote producer and not downloaded for this data audit. No model forward or new quality result.'}
    target=BASE/'additional_assets_v1_independent_data_audit.json'
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
