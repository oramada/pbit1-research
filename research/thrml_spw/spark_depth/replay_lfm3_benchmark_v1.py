"""Rebuild every LFM3 point with the SDK and replay prespecified benchmark windows."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys,time
import numpy as np
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT,point_hash
from .benchmark_v1 import atomic_json
sys.path.insert(0,str(Path('publish/pbit1/src').resolve()))
from pbit1 import FinitePBitProgram


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
 benchmark=OUT/'lfm3_benchmark_v2';p=json.loads((benchmark/'protocol.json').read_text());root=OUT/'lfm3_benchmark_replay_v1'
 protocol={'source_sha256':sha(__file__),'benchmark_protocol_sha256':sha(benchmark/'protocol.json'),'SDK_source_sha256':sha('publish/pbit1/src/pbit1/program.py'),
  'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),'windows':[0,303,606],'cases':p['cases'],
  'scope':'All seven model weight points independently reconstructed with SDK encoding/counts rather than producer sampler. All per-parameter count hashes and BF16 hashes checked. Three prespecified neural windows per point; not all607 forwards replayed.'}
 root.mkdir(exist_ok=True)
 if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
 else:atomic_json(root/'protocol.json',protocol)
 if args.freeze_only:print(json.dumps({'status':'replay_protocol_frozen_before_complete_benchmark','protocol_sha256':sha(root/'protocol.json')}));return
 audit=json.loads((OUT/'lfm3_benchmark_audit_v1/results.json').read_text());assert audit['result_sha256']==sha(benchmark/'results.json')
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
  params=dict(model.named_parameters());policies=p['selected_policies'];assert list(params)==[x['name']for x in policies]
  expected=Counter(n for n,m in model.named_modules()if any(True for _ in m.parameters(recurse=False)))
  data=OUT/'lfm3_data_v1'
  with np.load(data/'test_tokens.npz')as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
  windows=json.loads((data/'test_layout.json').read_text())['windows'];records=[]
  for case in p['cases']:
   folder=benchmark/case['name'];old=json.loads((folder/'parameters.json').read_text());checked=0
   with torch.no_grad(),safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as weights:
    for index,policy in enumerate(policies):
     raw=weights.get_tensor(policy['name'])
     if case['kind']=='original':value=raw
     else:
      program=FinitePBitProgram.encode(raw.float().numpy(),group_size=policy['selected_group'],layout=policy['selected_layout'])
      if case['kind']=='mean':value=torch.from_numpy(program.expectation()).to(torch.bfloat16)
      else:
       sample=program.reconstruct(32768,seed=case['seed'],parameter_index=index);value=torch.from_numpy(sample.weights).to(torch.bfloat16);ref=old['records'][index]
       assert sample.counts_sha256==ref['counts_sha256']and sample.counter_dtype==ref['counter_dtype']and sample.maximum_count==ref['max_count']
       assert hashlib.sha256(value.view(torch.uint16).numpy().tobytes()).hexdigest()==ref['BF16_sha256'];checked+=1
      del program
     params[policy['name']].copy_(value);del raw,value
     atomic_json(root/'progress.json',{'case':case['name'],'stage':'SDK_reconstruction','parameter':index,'elapsed_seconds':time.monotonic()-begin})
   point=point_hash(model);assert point==old['point_sha256'];replays=[]
   with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
    for index in protocol['windows']:
     w=windows[index];base=int(offsets[w['article']]);ids=tokens[base+w['context_start']:base+w['context_stop']];first=w['target_start']-w['context_start'];last=w['target_stop']-w['context_start'];n=last-first
     offloader.calls.clear();torch.cuda.reset_peak_memory_stats();out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False);assert Counter(offloader.calls)==expected
     scored=out.logits[:,first-1:last-1];target=torch.from_numpy(ids[first:last]).cuda();loss=np.empty(n,np.float64)
     for start in range(0,n,64):loss[start:start+64]=torch.nn.functional.cross_entropy(scored[0,start:start+64].float(),target[start:start+64],reduction='none').cpu().double().numpy()
     digest=hashlib.sha256(scored.cpu().view(torch.uint16).numpy().tobytes()).hexdigest();del out,scored,target
     file=folder/f'window{index:04d}.npz';receipt=json.loads(file.with_suffix('.json').read_text());assert digest==receipt['BF16_scored_logits_sha256']
     with np.load(file)as f:np.testing.assert_array_equal(loss,f['token_NLL'])
     assert torch.cuda.max_memory_allocated()<=2*1024**3;replays.append({'window':index,'targets':n,'logits_sha256':digest,'losses_bitwise_equal':True})
   del offloader
   record={'case':case,'point_sha256':point,'SDK_count_hashes_checked':checked,'replays':replays};records.append(record);atomic_json(root/(case['name']+'.json'),record);print('REPLAYED',case['name'],flush=True)
  atomic_json(root/'results.json',{'status':'all7_weight_points_and21_neural_windows_replayed','protocol_sha256':sha(root/'protocol.json'),'benchmark_result_sha256':sha(benchmark/'results.json'),'cases':records,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
 except Exception as error:atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
