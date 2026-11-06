"""Training-only full-model device-error and ideal calibration ceiling experiment."""
from pathlib import Path
import hashlib,json,time,math
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped
from .cpu_vector_validation_v1 import token_losses
from .benchmark_v1 import atomic_json

SEEDS=list(range(2026112401,2026112406))

def draw(raw,index,layout,kind,seed,torch):
 w=raw.float().numpy();a=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w;flat=a.reshape(-1);out=np.empty(flat.size,np.float32)
 devices=np.random.default_rng(np.random.SeedSequence([seed,index,11]));samples=np.random.default_rng(np.random.SeedSequence([seed,index,22]));ch=hashlib.sha256();dh=hashlib.sha256();bias_sse=variance_sse=observed=0.;clipped=0
 for start in range(0,len(flat),262144):
  q=flat[start:start+262144];code,lo,span,requested,var=mapped(q,128);fields=code.astype(np.float32)*np.float32(8/4095)-np.float32(4);target=2*fields.astype(np.float64)
  gain=devices.normal(1,.02,q.size);offset=devices.normal(0,.02,q.size);assert(gain>0).all();dh.update(gain.tobytes());dh.update(offset.tobytes())
  if kind=='ideal':actual_logit=target
  elif kind=='uncalibrated':actual_logit=gain*target+offset
  elif kind=='oracle':
   ideal_request=(target-offset)/gain/2;clipped+=int(np.count_nonzero(np.abs(ideal_request)>4))
   corrected=np.rint(np.clip((ideal_request+4)*(4095/8),0,4095)).astype(np.uint16)
   field=corrected.astype(np.float32)*np.float32(8/4095)-np.float32(4)
   actual_logit=gain*(2*field.astype(np.float64))+offset
  else:raise ValueError(kind)
  p=1/(1+np.exp(-actual_logit));low=np.repeat(lo,128)[:q.size].astype(np.float64);scale=np.repeat(span,128)[:q.size].astype(np.float64)
  mean=low+scale*p;counts=samples.binomial(32768,p).astype(np.uint16);ch.update(counts.tobytes());value=low+scale*counts.astype(np.float64)/32768;out[start:start+q.size]=value
  bias_sse+=float(np.sum((mean-requested)**2));variance_sse+=float(np.sum(scale**2*p*(1-p)/32768));observed+=float(np.sum((value-requested)**2))
 value=out.reshape(a.shape);value=value.T.copy()if layout=='transposed_matrix'and w.ndim==2 else value;rounded=torch.from_numpy(value.copy()).to(torch.bfloat16)
 return rounded,{'count_sha256':ch.hexdigest(),'device_characteristics_sha256':dh.hexdigest(),'bias_SSE_relative_to_requested_mean':bias_sse,'sampling_variance_SSE':variance_sse,'sample_SSE_relative_to_requested_mean':observed,'clipped_inverse_requests':clipped,'BF16_sha256':hashlib.sha256(rounded.view(torch.uint16).numpy().tobytes()).hexdigest()}

def main():
 import torch
 from transformers import AutoModelForCausalLM
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
 root=OUT/'gpt2_calibration_control_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
 prefs={x['weight']:x['preferred_by_local_variance']for x in json.loads((OUT/'activation_variance_v1/results.json').read_text())['operators']}
 protocol={'source_sha256':sha(__file__),'codec_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),'data_sha256':sha(OUT/'training_diagnostic_windows.npz'),'seeds':SEEDS,'gain_SD':.02,'logit_offset_SD':.02,'samples':32768,'group':128,'control_bits':12,'rows':list(range(8)),'tokens_per_row':129,'preferences':prefs,
 'scope':'FullGPT2, all parameters. FP32CPUarithmetic withBF16-roundedweights; 1024trainingtargets. Independent-binomialR1. Device errors illustrative, not measured hardware. Oracle correction knows true per-device gain/offset, re-quantizes the original12bit controls and clips to their range; it is an upper bound on calibration quality, not a free or deployable measured fix. TwoFP32 calibration values per device would require an additional8Nbytes if retained. No drift/cross-device correlation here.'}
 atomic_json(root/'protocol.json',protocol)
 try:
  model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval();params=dict(model.named_parameters());original={n:p.detach().to(torch.bfloat16).clone()for n,p in params.items()}
  with np.load(OUT/'training_diagnostic_windows.npz')as f:inputs=f['input_ids'][:8,:129].copy()
  summaries=[]
  for kind,seed in [('baseline',None)]+[(kind,s)for s in SEEDS for kind in ['ideal','uncalibrated','oracle']]:
   name=kind if seed is None else f'{kind}_seed{seed}';folder=root/name;folder.mkdir();records=[];point=hashlib.sha256()
   with torch.no_grad():
    for index,(key,param)in enumerate(params.items()):
     if kind=='baseline':value=original[key]
     else:value,record=draw(original[key],index,prefs.get(key,'storage_flat'),kind,seed,torch);records.append({'name':key,**record})
     point.update(key.encode());point.update(value.view(torch.uint16).numpy().tobytes());param.copy_(value.float())
   atomic_json(folder/'parameters.json',{'point_sha256':point.hexdigest(),'records':records});losses=[]
   for row,ids in enumerate(inputs):
    with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]),use_cache=False)
    loss=token_losses(out.logits.numpy(),ids);assert np.isfinite(loss).all();losses.extend(loss.tolist());del out
    with(folder/f'row{row}.npz').open('xb')as f:np.savez_compressed(f,token_NLL=loss,input_ids=ids)
    atomic_json(root/'progress.json',{'case':name,'row':row,'elapsed_seconds':time.monotonic()-begin})
   summary={'kind':kind,'seed':seed,'NLL':float(np.mean(losses)),'targets':len(losses),'point_sha256':point.hexdigest(),'bias_SSE_relative_to_requested_mean':sum(x['bias_SSE_relative_to_requested_mean']for x in records),'sampling_variance_SSE':sum(x['sampling_variance_SSE']for x in records),'clipped_inverse_requests':sum(x['clipped_inverse_requests']for x in records)};atomic_json(folder/'results.json',summary);summaries.append(summary);print(json.dumps(summary),flush=True)
  stats={};baseline=summaries[0]['NLL']
  for kind in ['ideal','uncalibrated','oracle']:
   values=np.array([x['NLL']for x in summaries if x['kind']==kind])-baseline;half=2.7764451051977987*values.std(ddof=1)/np.sqrt(5);stats[kind]={'PPL_ratio':float(np.exp(values.mean())),'MC95_t4':np.exp([values.mean()-half,values.mean()+half]).tolist()}
  assert not torch.cuda.is_initialized()
  atomic_json(root/'results.json',{'status':'all16_training_cases_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'cases':summaries,'statistics':stats,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
 except Exception as error:atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
