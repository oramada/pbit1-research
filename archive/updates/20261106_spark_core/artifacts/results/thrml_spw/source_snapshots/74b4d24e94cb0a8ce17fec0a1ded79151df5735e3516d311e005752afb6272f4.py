"""CPU-only pinned GPT2 assets and training-only diagnostic windows."""
from pathlib import Path
import datetime,hashlib,json,shutil,time
import numpy as np

OUT=Path('results/thrml_spw/spark_depth_20261106')
CHECKPOINT=Path('data/thrml_spw/spark_depth_20261106/gpt2')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for data in iter(lambda:f.read(16*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from research.thrml_spw.broad.data_layout_v1 import articles_from_rows
    OUT.mkdir(parents=True,exist_ok=True);CHECKPOINT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'preparation_protocol.json'
    inventory=Path('results/thrml_spw/broad_verification_20261105/additional_model_inventory/pinned_hub_inventory.json')
    record=next(r for r in json.loads(inventory.read_text())['records']if r['id']=='openai-community/gpt2')
    train=Path('data/thrml_spw/lfm3_1p7b/adaptation_parquet/train.parquet')
    source_manifest=Path('results/thrml_spw/lfm3_1p7b/adaptation_data/manifest.json')
    p={'status':'training_only_diagnostic_preparation_frozen_before_new_model_forwards','recorded_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256':sha(__file__),'model':record['id'],'revision':record['revision'],'inventory_sha256':sha(inventory),
        'training_parquet':str(train),'training_parquet_sha256':sha(train),'training_origin_manifest_sha256':sha(source_manifest),
        'selection':'Parse complete training articles using existing pinned parser; permutearticleindiceswithseed2026110601; takefirst32eligiblearticleswithatleast257GPT2tokens. First8articles provide128target diagnosticwindows; next8provide128target calibrationchecks; final16trainingarticles reserved for candidateconfirmation. No test/validationforwards or labelsused.',
        'selection_key':2026110601,'windows':32,'window_tokens':257,'initial_diagnostic_targets':1024,
        'all_learned_parameters_in_final_representation':True,'diagnostic_ablation_policy':'Selective deterministicrestoration allowed onlyasexplicitcausaldiagnostics, nevercountedasallonepbitresult.',
        'GPU_policy':'LocalDGXSparkonly; do notstopunrelateduca32workloadwithoutuserapproval; optionalsmallmemorycappedsharing. PreparationisCPUonly.',
        'cloud_policy':'No Modal apps/functions/jobs may be started. Read-only recovery of preserved evidence permitted.',
        'scope':'Researchpreparation, notneuralqualityresult or proof of universal numericalprecision.'}
    if protocol.exists():assert json.loads(protocol.read_text())==p,'Existing preparation contract differs; use a new version'
    else:
        with protocol.open('x')as f:json.dump(p,f,indent=2);f.write('\n')
    assert shutil.disk_usage('.').free>4*1024**3,'Insufficient disk headroom'
    files={}
    for item in record['files']:
        local=CHECKPOINT/item['path']
        if not local.exists():local=Path(hf_hub_download(record['id'],item['path'],revision=record['revision'],local_dir=CHECKPOINT))
        assert local.stat().st_size==item['bytes']
        digest=sha(local)
        if item['lfs_sha256']:assert digest==item['lfs_sha256']
        else:assert hashlib.sha1(b'blob '+str(local.stat().st_size).encode()+b'\0'+local.read_bytes()).hexdigest()==item['git_blob_id']
        files[str(local)]={'sha256':digest,'bytes':local.stat().st_size}
        print(json.dumps({'verified_asset':str(local),'bytes':local.stat().st_size}),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
    rows=pq.read_table(train,columns=['text'])['text'].to_pylist();articles=articles_from_rows(rows)
    order=np.random.default_rng(p['selection_key']).permutation(len(articles));chosen=[];tokens=[]
    for index in order:
        article=articles[int(index)];ids=tokenizer(article['text'],add_special_tokens=False)['input_ids']
        if len(ids)<257:continue
        chosen.append({'training_article_index':int(index),'source_row_start':article['source_row_start'],
            'text_sha256':hashlib.sha256(article['text'].encode()).hexdigest(),'complete_token_count':len(ids)})
        tokens.append(ids[:257])
        if len(tokens)==32:break
    assert len(tokens)==32 and len({r['training_article_index']for r in chosen})==32
    arrays=OUT/'training_diagnostic_windows.npz'
    with arrays.open('xb')as f:np.savez_compressed(f,input_ids=np.array(tokens,np.int64))
    result={'status':'pinned_GPT2_and_disjoint_training_article_windows_ready_without_model_forwards',
        'protocol_sha256':sha(protocol),'files':files,'training_windows':chosen,'arrays':str(arrays),'arrays_sha256':sha(arrays),
        'diagnostic_article_rows':list(range(8)),'calibration_check_rows':list(range(8,16)),'candidate_confirmation_rows':list(range(16,32)),
        'source_sha256':sha(__file__),'checkpoint':str(CHECKPOINT),'scope':'CPUasset/tokenizationpreparationonly. Rawlocaltrainingtextre-tokenizedwithGPT2tokenizer; no reuseofLFMtoken IDs or heldout testlabels.'}
    with (OUT/'prepared.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'articles':32,'arrays_sha256':result['arrays_sha256']}),flush=True)

if __name__=='__main__':main()
