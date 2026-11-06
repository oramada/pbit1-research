"""CPU training-only, full-GPT2 variance-matched Gaussian negative control."""
from pathlib import Path
import json,math,time,hashlib
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped
from .counter_sampler_v3 import sample_parameter
from .cpu_vector_validation_v1 import token_losses
from .benchmark_v1 import atomic_json

SEEDS=list(range(2026112201,2026112206))

def gaussian(raw,index,layout,torch,seed):
 w=raw.float().numpy();a=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w;flat=a.reshape(-1);result=np.empty(flat.size,np.float32)
 rng=np.random.default_rng(np.random.SeedSequence([seed,index]));sse=expected=0.;h=hashlib.sha256()
 for start in range(0,flat.size,262144):
  q=flat[start:start+262144];mean,var=mapped(q,128)[3:];noise=rng.normal(size=q.size)*np.sqrt(var/32768);values=mean+noise
  result[start:start+q.size]=values;sse+=float(noise@noise);expected+=float(np.sum(var/32768));h.update(noise.tobytes())
 values=result.reshape(a.shape)
 if layout=='transposed_matrix'and w.ndim==2:values=values.T.copy()
 rounded=torch.from_numpy(values.copy()).to(torch.bfloat16)
 return rounded,{'normal_noise_sha256':h.hexdigest(),'pre_FP32_SSE_about_mean':sse,'expected_SSE':expected,'BF16_sha256':hashlib.sha256(rounded.view(torch.uint16).numpy().tobytes()).hexdigest()}

def main():
 import torch
 from transformers import AutoModelForCausalLM
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
 prefs={x['weight']:x['preferred_by_local_variance']for x in json.loads((OUT/'activation_variance_v1/results.json').read_text())['operators']}
 root=OUT/'gaussian_control_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
 protocol={'source_sha256':sha(__file__),'counter_sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v3.py')),'codec_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),
  'data_sha256':sha(OUT/'training_diagnostic_windows.npz'),'checkpoint_sha256':sha(CHECKPOINT/'model.safetensors'),'preferences':prefs,'seeds':SEEDS,
  'rows':list(range(8)),'length':129,'samples':32768,'group':128,'Gaussian_seed_offset':1000,
  'scope':'Training-only GPT2 diagnostic. FP32 CPU arithmetic using BF16-rounded original and reconstructed weights. Full-model Gaussian noise matched per coordinate to the finite-controller mean and binomial variance before FP32/BF16 rounding. Independent Gaussian and Bernoulli random keys. No GPU, test-data evaluation, distributional equivalence, or hardware claim.'}
 atomic_json(root/'protocol.json',protocol)
 try:
  model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval()
  params=dict(model.named_parameters());original={n:p.detach().to(torch.bfloat16).clone()for n,p in params.items()};assert sum(p.numel()for p in params.values())==124439808
  with np.load(OUT/'training_diagnostic_windows.npz')as f:inputs=f['input_ids'][:8,:129].copy()
  cases=[('BF16',None)]+[(kind,seed)for seed in SEEDS for kind in ['pbit','Gaussian']];summaries=[]
  for kind,seed in cases:
   name=kind if seed is None else f'{kind}_seed{seed}';folder=root/name;folder.mkdir();records=[];point=hashlib.sha256()
   with torch.no_grad():
    for index,(key,p)in enumerate(params.items()):
     layout=prefs.get(key,'storage_flat')
     if kind=='BF16':value=original[key]
     elif kind=='pbit':value,record=sample_parameter(original[key],index,layout,128,32768,torch,seed);records.append({'name':key,**record})
     else:value,record=gaussian(original[key],index,layout,torch,seed+1000);records.append({'name':key,**record})
     point.update(key.encode());point.update(value.view(torch.uint16).numpy().tobytes());p.copy_(value.float())
   atomic_json(folder/'parameters.json',{'point_sha256':point.hexdigest(),'records':records});losses=[]
   for row,ids in enumerate(inputs):
    with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]),use_cache=False)
    loss=token_losses(out.logits.numpy(),ids);assert np.isfinite(loss).all();losses.extend(loss.tolist());del out
    with(folder/f'row{row}.npz').open('xb')as f:np.savez_compressed(f,token_NLL=loss,input_ids=ids)
    atomic_json(root/'progress.json',{'case':name,'row':row,'elapsed_seconds':time.monotonic()-begin})
   result={'kind':kind,'seed':seed,'NLL':float(np.mean(losses)),'targets':len(losses),'point_sha256':point.hexdigest()};atomic_json(folder/'results.json',result);summaries.append(result);print(json.dumps(result),flush=True)
  base=summaries[0]['NLL'];stats={}
  for kind in ['pbit','Gaussian']:
   v=np.array([x['NLL']for x in summaries if x['kind']==kind]);half=2.7764451051977987*v.std(ddof=1)/np.sqrt(5)
   stats[kind]={'PPL_ratio_to_baseline':float(np.exp(v.mean()-base)),'MC95_t4_interval':np.exp([v.mean()-base-half,v.mean()-base+half]).tolist()}
  assert not torch.cuda.is_initialized()
  atomic_json(root/'results.json',{'status':'all11_fullmodel_training_controls_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'cases':summaries,'statistics':stats,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
 except Exception as error:
  atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
