"""Frozen complete LFM3 WikiText2 evaluation after training-cohort qualification."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,sys,time
import numpy as np
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT,point_hash
from .counter_sampler_v3 import sample_parameter
from .lfm3_training_depth_v1 import mean_parameter
from .benchmark_v1 import atomic_json

SEEDS=list(range(2026111901,2026111906))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    calibration=OUT/'lfm3_calibration_v2/results.json';audit=OUT/'lfm3_calibration_v3_audit.json';data=OUT/'lfm3_data_v1';root=OUT/'lfm3_benchmark_v1'
    assert json.loads(audit.read_text())['result_sha256']==sha(calibration)
    policies=json.loads(calibration.read_text())['policies'];meta=json.loads((data/'results.json').read_text())
    for f,key in [('test_tokens.npz','test_tokens_sha256'),('test_layout.json','test_layout_sha256')]:assert sha(data/f)==meta[key]
    cases=[{'name':'BF16','kind':'original'},{'name':'selected_mean','kind':'mean'}]+[{'name':f'selected_seed{s}','kind':'selected','seed':s}for s in SEEDS]
    protocol={'source_sha256':sha(__file__),'calibration_sha256':sha(calibration),'calibration_audit_sha256':sha(audit),'sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v3.py')),
      'mean_source_sha256':sha(Path(__file__).with_name('lfm3_training_depth_v1.py')),'codec_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),
      'test_tokens_sha256':meta['test_tokens_sha256'],'test_layout_sha256':meta['test_layout_sha256'],'cases':cases,'targets':294442,'articles':62,'windows':607,
      'parameters':1728801536,'samples':32768,'control_bits':12,'selected_policies':policies,'GPU_cap_bytes':2*1024**3,
      'training_gate':'Before heldout scoring, complete five-seed training cohort and its independent score audit; require upper95%t4 PPLratio<=1.02. Otherwise retain failed calibration and choose a new version.',
      'primary_gate':'All five fresh stochastic seeds and both controls on all294442targets. Upper endpoint95%t4 CI on seed-level mean deltaNLL relative to originalBF16<=ln1.02.',
      'logits':'All hidden states in context, head computes only scored prediction positions via logits_to_keep; CE in64target chunks, no KVcache.',
      'scope':'Previously inspected WikiText2 test with LFM3 tokenizer; not newly unexposed data. Fresh draw seeds; ideal independent binomial R1, all learned scalars K1, grouped finite12bit controls. CPU-master BF16 neural execution; no fullmodel nativeTHRML, physical hardware, or losslessBF16 claim.'}
    root.mkdir(exist_ok=True)
    if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:atomic_json(root/'protocol.json',protocol)
    if args.freeze_only:print(json.dumps({'status':'prospective_protocol_frozen','protocol_sha256':sha(root/'protocol.json')}));return
    training=json.loads((OUT/'lfm3_training_depth_v1/results.json').read_text());training_audit=json.loads((OUT/'lfm3_training_depth_analysis_v1/results.json').read_text())
    assert training_audit['results_sha256']==sha(OUT/'lfm3_training_depth_v1/results.json')and training['MC95_t4_PPL_ratio_interval'][1]<=1.02,'Training qualification missing or failed; do not score heldout.'
    import fcntl
    lock=(OUT/'gpu_research.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name();torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not(root/'failure.json').exists()
    if(root/'results.json').exists():return
    begin=time.monotonic()
    try:
        model=AutoModelForCausalLM.from_pretrained(ROOT,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
        sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *a:False
        params=dict(model.named_parameters());assert list(params)==[p['name']for p in policies]
        expected=Counter(n for n,m in model.named_modules()if any(True for _ in m.parameters(recurse=False)))
        with np.load(data/'test_tokens.npz')as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
        windows=json.loads((data/'test_layout.json').read_text())['windows'];assert len(windows)==607
        summaries=[]
        for case in cases:
            folder=root/case['name'];folder.mkdir(exist_ok=True);records=[]
            with torch.no_grad(),safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as weights:
                for index,p in enumerate(policies):
                    raw=weights.get_tensor(p['name'])
                    if case['kind']=='original':value=raw
                    elif case['kind']=='mean':value=mean_parameter(raw,p['selected_layout'],p['selected_group'],torch)
                    else:
                        value,record=sample_parameter(raw,index,p['selected_layout'],p['selected_group'],32768,torch,case['seed']);record['name']=p['name'];records.append(record)
                    params[p['name']].copy_(value);del raw,value
            point=point_hash(model)
            if case['kind']=='original':assert point==json.loads((OUT/'lfm3_full_preflight_v2_audit.json').read_text())['weight_point_sha256']
            receipt={'point_sha256':point,'records':records}
            if(folder/'parameters.json').exists():assert json.loads((folder/'parameters.json').read_text())==receipt
            else:atomic_json(folder/'parameters.json',receipt)
            total=0.;count=0;sums=np.zeros(62);counts=np.zeros(62,dtype=np.int64);peak=0
            with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
                for index,window in enumerate(windows):
                    first=window['target_start']-window['context_start'];last=window['target_stop']-window['context_start'];n=last-first
                    base=int(offsets[window['article']]);ids=tokens[base+window['context_start']:base+window['context_stop']].copy();input_hash=hashlib.sha256(ids.tobytes()).hexdigest()
                    file=folder/f'window{index:04d}.npz';record_file=file.with_suffix('.json')
                    if record_file.exists():
                        saved=json.loads(record_file.read_text());assert saved['point_sha256']==point and saved['input_sha256']==input_hash and saved['arrays_sha256']==sha(file)
                        with np.load(file)as f:loss=f['token_NLL'].copy()
                    else:
                        offloader.calls.clear();torch.cuda.reset_peak_memory_stats()
                        out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False,logits_to_keep=torch.arange(first-1,last-1,device='cuda'))
                        assert Counter(offloader.calls)==expected and offloader.active_bytes==0 and out.logits.shape==(1,n,128000)
                        targets=torch.from_numpy(ids[first:last]).cuda();loss=np.empty(n,np.float64)
                        for start in range(0,n,64):loss[start:start+64]=torch.nn.functional.cross_entropy(out.logits[0,start:start+64].float(),targets[start:start+64],reduction='none').cpu().double().numpy()
                        logit_sha=hashlib.sha256(out.logits.cpu().view(torch.uint16).numpy().tobytes()).hexdigest();del out,targets
                        temp=file.with_suffix('.tmp')
                        with temp.open('wb')as f:np.savez_compressed(f,token_NLL=loss)
                        temp.replace(file)
                        saved={'window':window,'point_sha256':point,'input_sha256':input_hash,'arrays_sha256':sha(file),'BF16_scored_logits_sha256':logit_sha,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()};atomic_json(record_file,saved)
                    assert loss.shape==(n,)and np.isfinite(loss).all();total+=float(loss.sum());count+=n;sums[window['article']]+=float(loss.sum());counts[window['article']]+=n;peak=max(peak,saved['GPU_peak_allocated_bytes']);assert peak<=2*1024**3
                    atomic_json(root/'progress.json',{'case':case['name'],'window':index,'windows':607,'scored_targets':count,'elapsed_seconds':time.monotonic()-begin})
            del offloader
            assert count==294442
            summary={'case':case,'NLL':total/count,'PPL':math.exp(total/count),'targets':count,'article_NLL_sums':sums.tolist(),'article_target_counts':counts.tolist(),'point_sha256':point,'GPU_peak_allocated_bytes':peak}
            atomic_json(folder/'results.json',summary);summaries.append(summary);print(json.dumps(summary),flush=True)
        delta=np.array([x['NLL']for x in summaries[2:]])-summaries[0]['NLL'];half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
        atomic_json(root/'results.json',{'status':'all7cases_complete_pending_independent_audit','protocol_sha256':sha(root/'protocol.json'),'training_qualification_sha256':sha(OUT/'lfm3_training_depth_analysis_v1/results.json'),'cases':summaries,'PPL_ratio':float(np.exp(delta.mean())),'MC95_t4_PPL_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'pass':bool(delta.mean()+half<=np.log(1.02)),'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']})
    except Exception as error:
        atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
