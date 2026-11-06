"""Pinned cross-architecture assets and IMDb training calibration, CPU only."""
from pathlib import Path
import json,hashlib
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    inventory=Path('results/thrml_spw/broad_verification_20261105/additional_model_inventory/pinned_hub_inventory.json');records=json.loads(inventory.read_text())['records']
    model=next(r for r in records if r['id']=='distilbert/distilbert-base-uncased-finetuned-sst-2-english');dataset=next(r for r in records if r['id']=='stanfordnlp/imdb')
    root=OUT/'distilbert_preparation_v1';root.mkdir(exist_ok=False);checkpoint=Path('data/thrml_spw/spark_depth_20261106/distilbert');checkpoint.mkdir(parents=True,exist_ok=True)
    (root/'protocol.json').write_text(json.dumps({'source_sha256':sha(__file__),'inventory_sha256':sha(inventory),'model':model['id'],'model_revision':model['revision'],'dataset_revision':dataset['revision'],'training_review_selection_seed':2026110901,'calibration_reviews':64,'token_length':256,'scope':'Trainpartitioncalibrationonly. No model forwards or benchmarklabelselection.'},indent=2)+'\n')
    files={}
    for item in model['files']:
        local=Path(hf_hub_download(model['id'],item['path'],revision=model['revision'],local_dir=checkpoint));assert local.stat().st_size==item['bytes'];digest=sha(local)
        if item['lfs_sha256']:assert digest==item['lfs_sha256']
        else:assert hashlib.sha1(b'blob '+str(local.stat().st_size).encode()+b'\0'+local.read_bytes()).hexdigest()==item['git_blob_id']
        files[str(local)]=digest
    path=Path(hf_hub_download(dataset['id'],'plain_text/train-00000-of-00001.parquet',repo_type='dataset',revision=dataset['revision'],local_dir='data/thrml_spw/spark_depth_20261106/imdb'))
    table=pq.read_table(path,columns=['text','label']);order=np.random.default_rng(2026110901).permutation(len(table))[:64];texts=[table['text'][int(i)].as_py()for i in order];labels=np.array([table['label'][int(i)].as_py()for i in order],np.int64)
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True);enc=tok(texts,padding='max_length',truncation=True,max_length=256,return_tensors='np')
    with(root/'calibration.npz').open('xb')as f:np.savez_compressed(f,input_ids=enc['input_ids'],attention_mask=enc['attention_mask'],labels=labels,source_review_indices=order)
    (root/'results.json').write_text(json.dumps({'status':'pinned_distilbert_and64training_reviews_ready_no_forwards','checkpoint':str(checkpoint),'files':files,'training_file':str(path),'training_file_sha256':sha(path),'calibration_sha256':sha(root/'calibration.npz'),'review_text_sha256':[hashlib.sha256(t.encode()).hexdigest()for t in texts]},indent=2)+'\n');print('Pinned DistilBERT and IMDb training calibration ready')

if __name__=='__main__':main()
