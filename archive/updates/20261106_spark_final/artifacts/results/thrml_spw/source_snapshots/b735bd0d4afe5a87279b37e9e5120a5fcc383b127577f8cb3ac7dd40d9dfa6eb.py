"""Complete prospective GPT2 validation of the lower-work K1 allocation."""
from pathlib import Path
import argparse,hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .benchmark_v1 import atomic_json
from .counter_sampler_v3 import sample_parameter


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
 planfile=OUT/'reduced_work_GPU_plan_v1.json';plan=json.loads(planfile.read_text());root=OUT/'reduced_work_gpu_v1';data=plan['complete_benchmark'];policies=plan['parameters']
 assert sum(x['parameters']for x in policies)==124439808 and sum(x['parameters']*x['samples']for x in policies)==plan['nominal_observations_per_draw']==2378245865472
 assert max(x['samples']for x in policies)==32768 and plan['observation_fraction_of_uniform32768']<=.60
 for key in ['tokens','layout']:assert sha(data[key])==data[key+'_sha256']
 cases=[{'name':'BF16','kind':'original'}]+[{'name':f'reduced_seed{s}','kind':'reduced','seed':s}for s in plan['seeds']]
 protocol={'source_sha256':sha(__file__),'plan_sha256':sha(planfile),'sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v3.py')),
  'checkpoint_sha256':sha(CHECKPOINT/'model.safetensors'),'cases':cases,'data':data,'policies':policies,'GPU_cap_bytes':2*1024**3,
  'primary_gate':plan['primary_gate'],'observation_fraction':plan['observation_fraction_of_uniform32768'],'extra_metadata_bytes':53088,'scope':plan['scope']+' R1 independent-binomial reconstruction reused across examples, BF16 GPU arithmetic.'}
 root.mkdir(exist_ok=True)
 if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
 else:atomic_json(root/'protocol.json',protocol)
 if args.freeze_only:print(json.dumps({'status':'full_lower_work_GPU_runner_frozen','protocol_sha256':sha(root/'protocol.json')}));return
 import fcntl
 lock=(OUT/'gpu_research.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 from transformers import AutoModelForCausalLM
 from research.thrml_spw.broad.r1_model_v1 import observations
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name();torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert not(root/'failure.json').exists()
 if(root/'results.json').exists():return
 begin=time.monotonic()
 try:
  model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
  params=dict(model.named_parameters());assert list(params)==[x['name']for x in policies]
  original={n:p.detach().cpu().clone()for n,p in params.items()};assert model.lm_head.weight is model.transformer.wte.weight
  with np.load(data['tokens'])as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
  windows=json.loads(Path(data['layout']).read_text())['windows'];assert len(windows)==586
  summaries=[]
  for case in cases:
   folder=root/case['name'];folder.mkdir(exist_ok=True);digest=hashlib.sha256();records=[]
   with torch.no_grad():
    for index,policy in enumerate(policies):
     name=policy['name']
     if case['kind']=='original':value=original[name]
     else:
      value,record=sample_parameter(original[name],index,policy['layout'],policy['group'],policy['samples'],torch,case['seed']);records.append({'name':name,**record})
     digest.update(name.encode());digest.update(value.view(torch.uint16).numpy().tobytes());params[name].copy_(value.cuda())
   point=digest.hexdigest();receipt={'point_sha256':point,'records':records}
   if(folder/'parameters.json').exists():assert json.loads((folder/'parameters.json').read_text())==receipt
   else:atomic_json(folder/'parameters.json',receipt)
   total=0.;count=0;sums=np.zeros(62);counts=np.zeros(62,np.int64);peak=0
   for index,w in enumerate(windows):
    base=int(offsets[w['article']]);ids=tokens[base+w['context_start']:base+w['context_stop']][None];input_hash=hashlib.sha256(ids.tobytes()).hexdigest();file=folder/f'window{index:04d}.npz';rfile=file.with_suffix('.json')
    if rfile.exists():
     r=json.loads(rfile.read_text());assert r['point_sha256']==point and r['input_sha256']==input_hash and r['arrays_sha256']==sha(file)
     with np.load(file)as f:loss=f['token_NLL'].copy()
    else:
     torch.cuda.reset_peak_memory_stats()
     with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids).cuda(),use_cache=False)
     logits=out.logits.float().cpu().numpy();obs=observations({'task':'gpt2',**w},ids,None,logits);loss=obs['token_NLL'];temp=file.with_suffix('.tmp')
     with temp.open('wb')as f:np.savez_compressed(f,**obs)
     temp.replace(file);r={'window':w,'point_sha256':point,'input_sha256':input_hash,'arrays_sha256':sha(file),'logits_sha256':hashlib.sha256(logits.tobytes()).hexdigest(),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()};atomic_json(rfile,r);del out,logits,obs
    assert loss.shape==(w['target_stop']-w['target_start'],)and np.isfinite(loss).all();total+=float(loss.sum());count+=len(loss);sums[w['article']]+=float(loss.sum());counts[w['article']]+=len(loss);peak=max(peak,r['GPU_peak_allocated_bytes']);assert peak<=2*1024**3
    atomic_json(root/'progress.json',{'case':case['name'],'window':index,'scored_targets':count,'elapsed_seconds':time.monotonic()-begin})
   assert count==283225
   summary={'case':case,'NLL':total/count,'PPL':math.exp(total/count),'targets':count,'article_NLL_sums':sums.tolist(),'article_target_counts':counts.tolist(),'point_sha256':point,'GPU_peak_allocated_bytes':peak};atomic_json(folder/'results.json',summary);summaries.append(summary);print(json.dumps(summary),flush=True)
  values=np.array([x['NLL']for x in summaries[1:]])-summaries[0]['NLL'];half=2.7764451051977987*values.std(ddof=1)/np.sqrt(5)
  atomic_json(root/'results.json',{'status':'all6_lower_work_cases_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'cases':summaries,'PPL_ratio':float(np.exp(values.mean())),'MC95_t4_PPL_ratio_interval':np.exp([values.mean()-half,values.mean()+half]).tolist(),'pass':bool(values.mean()+half<=np.log(1.02)),'observation_fraction':plan['observation_fraction_of_uniform32768'],'scope':protocol['scope']})
 except Exception as error:atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
