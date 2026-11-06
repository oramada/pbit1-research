"""SDK count/weight reconstruction and fixed neural replay for the lower-work policy."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .benchmark_v1 import atomic_json
sys.path.insert(0,str(Path('publish/pbit1/src').resolve()))
from pbit1 import FinitePBitProgram


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args();root=OUT/'reduced_work_replay_v1';benchmark=OUT/'reduced_work_gpu_v1';p=json.loads((benchmark/'protocol.json').read_text())
 protocol={'source_sha256':sha(__file__),'benchmark_protocol_sha256':sha(benchmark/'protocol.json'),'SDK_source_sha256':sha('publish/pbit1/src/pbit1/program.py'),'windows':[0,293,585],'cases':p['cases'],'scope':'All6modelpoints reconstructed;5x148counts checked through independentSDK. Threeprespecifiedfullnetworkwindows perpoint, notall586forwards.'}
 root.mkdir(exist_ok=True)
 if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
 else:atomic_json(root/'protocol.json',protocol)
 if args.freeze_only:print(json.dumps({'status':'lower_work_replay_protocol_frozen'}));return
 assert json.loads((OUT/'reduced_work_gpu_v1_audit.json').read_text())['result_sha256']==sha(benchmark/'results.json')
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
  model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval();params=dict(model.named_parameters());original={n:x.detach().cpu().clone()for n,x in params.items()}
  with np.load(p['data']['tokens'])as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
  windows=json.loads(Path(p['data']['layout']).read_text())['windows'];records=[]
  for case in p['cases']:
   folder=benchmark/case['name'];old=json.loads((folder/'parameters.json').read_text());point=hashlib.sha256();checked=0
   with torch.no_grad():
    for index,policy in enumerate(p['policies']):
     name=policy['name']
     if case['kind']=='original':value=original[name]
     else:
      program=FinitePBitProgram.encode(original[name].float().numpy(),group_size=policy['group'],layout=policy['layout']);sample=program.reconstruct(policy['samples'],seed=case['seed'],parameter_index=index);value=torch.from_numpy(sample.weights).to(torch.bfloat16);ref=old['records'][index]
      assert sample.counts_sha256==ref['counts_sha256']and sample.maximum_count==ref['max_count'];assert hashlib.sha256(value.view(torch.uint16).numpy().tobytes()).hexdigest()==ref['BF16_sha256'];checked+=1
     point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes());params[name].copy_(value.cuda())
   assert point.hexdigest()==old['point_sha256'];replays=[]
   for wi in protocol['windows']:
    w=windows[wi];start=int(offsets[w['article']]);ids=tokens[start+w['context_start']:start+w['context_stop']][None]
    with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids).cuda(),use_cache=False)
    logits=out.logits.float().cpu().numpy();obs=observations({'task':'gpt2',**w},ids,None,logits);receipt=json.loads((folder/f'window{wi:04d}.json').read_text());assert hashlib.sha256(logits.tobytes()).hexdigest()==receipt['logits_sha256']
    with np.load(folder/f'window{wi:04d}.npz')as f:
     for key in obs:np.testing.assert_array_equal(obs[key],f[key])
    replays.append({'window':wi,'all_outputs_bitwise_equal':True});del out,logits
   record={'case':case,'point_sha256':point.hexdigest(),'SDK_count_hashes_checked':checked,'replays':replays};records.append(record);atomic_json(root/(case['name']+'.json'),record);print('REPLAYED',case['name'],flush=True)
  atomic_json(root/'results.json',{'status':'all6_weightpoints_and18_neural_windows_replayed','protocol_sha256':sha(root/'protocol.json'),'benchmark_result_sha256':sha(benchmark/'results.json'),'cases':records,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
 except Exception as error:atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
