"""Rescore all negative-control outputs and independently check variance matching."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha

def main():
 root=OUT/'gaussian_control_v1';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
 stats={};nlls={};moment_rows=[]
 for case in r['cases']:
  name=case['kind']if case['seed']is None else f"{case['kind']}_seed{case['seed']}";losses=[]
  for row in range(8):
   with np.load(root/name/f'row{row}.npz')as f:assert f['token_NLL'].shape==(128,)and f['input_ids'].shape==(129,);losses.extend(f['token_NLL'])
  nll=float(np.mean(losses));assert nll==case['NLL'];nlls.setdefault(case['kind'],[]).append(nll)
 for kind in ['pbit','Gaussian']:
  values=np.asarray(nlls[kind]);assert len(values)==5
  delta=values-nlls['BF16'][0];half=2.7764451051977987*values.std(ddof=1)/np.sqrt(5)
  stats[kind]={'PPL_ratio_to_baseline':float(np.exp(delta.mean())),'MC95_t4_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist()}
  assert stats[kind]==r['statistics'][kind]
 for seed in p['seeds']:
  b=json.loads((root/f'pbit_seed{seed}/parameters.json').read_text())['records'];g=json.loads((root/f'Gaussian_seed{seed}/parameters.json').read_text())['records'];assert len(b)==len(g)==148
  for a,c in zip(b,g):assert a['name']==c['name']and a['expected_SSE']==c['expected_SSE']
  moment_rows.append({'seed':seed,'expected_SSE':sum(a['expected_SSE']for a in b),'binomial_SSE':sum(a['SSE_about_mean']for a in b),'Gaussian_SSE':sum(a['pre_FP32_SSE_about_mean']for a in g)})
 # Independent one-shot reconstruction, without the producer's chunk loop or mapped codec.
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2);checks=[]
 seed=p['seeds'][0]
 records=json.loads((root/f'Gaussian_seed{seed}/parameters.json').read_text())['records'];lookup={x['name']:(i,x)for i,x in enumerate(records)}
 with safe_open(str(CHECKPOINT/'model.safetensors'),framework='pt',device='cpu')as weights:
  for name in ['transformer.h.0.ln_1.weight','transformer.h.0.attn.c_proj.weight','transformer.ln_f.bias']:
   index,record=lookup[name];w=weights.get_tensor(name).to(torch.bfloat16).float().numpy();transposed=p['preferences'].get(name,'storage_flat')=='transposed_matrix'and w.ndim==2
   a=w.T.copy()if transposed else w;flat=a.reshape(-1);groups=np.pad(flat,(0,(-flat.size)%128),constant_values=np.nan).reshape(-1,128)
   lo=np.nanmin(groups,axis=1);span=np.nanmax(groups,axis=1)-lo;unit=np.divide(groups.astype(np.float64)-lo[:,None],span[:,None],out=np.full(groups.shape,.5),where=span[:,None]!=0).reshape(-1)[:flat.size]
   prob=np.clip(unit,1/(1+np.exp(8)),1-1/(1+np.exp(8)));control=np.rint(np.clip((.5*(np.log(prob)-np.log1p(-prob))+4)*(4095/8),0,4095)).astype(np.uint16)
   field=control.astype(np.float32)*np.float32(8/4095)-np.float32(4);actual=1/(1+np.exp(-2*field.astype(np.float64)))
   low=np.repeat(lo,128)[:flat.size].astype(np.float64);scale=np.repeat(span,128)[:flat.size].astype(np.float64);mean=low+scale*actual;variance=scale**2*actual*(1-actual)/32768
   noise=np.random.default_rng(np.random.SeedSequence([seed+1000,index])).normal(size=flat.size)*np.sqrt(variance)
   assert hashlib.sha256(noise.tobytes()).hexdigest()==record['normal_noise_sha256']
   value=(mean+noise).astype(np.float32).reshape(a.shape);value=value.T.copy()if transposed else value
   rounded=torch.from_numpy(value.copy()).to(torch.bfloat16)
   assert hashlib.sha256(rounded.view(torch.uint16).numpy().tobytes()).hexdigest()==record['BF16_sha256']
   checks.append(name)
 output={'status':'all11_scores_740_paired_expected_variances_and_three_Gaussian_replays_passed','result_sha256':sha(root/'results.json'),'source_sha256':sha(__file__),'statistics':stats,'moment_rows':moment_rows,'independent_replays':checks,'scope':p['scope']}
 (OUT/'gaussian_control_v1_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
if __name__=='__main__':main()
