"""Audit full-model device-error controls, including independent SDK-based replays."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
sys.path.insert(0,str(Path('publish/pbit1/src').resolve()))
from pbit1 import FinitePBitProgram


def main():
 root=OUT/'gpt2_calibration_control_v1';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
 expected_names=[x['name']for x in json.loads((OUT/'weight_survey_v1/results.json').read_text())['parameters']]
 assert len(r['cases'])==16 and {(x['kind'],x['seed'])for x in r['cases']}=={('baseline',None)}|{(k,s)for s in p['seeds']for k in ['ideal','uncalibrated','oracle']}
 vals={};paired=[]
 for case in r['cases']:
  name=case['kind']if case['seed']is None else f"{case['kind']}_seed{case['seed']}";loss=[]
  for row in range(8):
   with np.load(root/name/f'row{row}.npz')as f:assert f['token_NLL'].shape==(128,)and f['input_ids'].shape==(129,)and np.isfinite(f['token_NLL']).all();loss.extend(f['token_NLL'])
  np.testing.assert_allclose(np.mean(loss),case['NLL'],rtol=1e-14);vals.setdefault(case['kind'],[]).append(case['NLL'])
 for seed in p['seeds']:
  data={kind:json.loads((root/f'{kind}_seed{seed}/parameters.json').read_text())['records']for kind in ['ideal','uncalibrated','oracle']}
  for records in data.values():assert [x['name']for x in records]==expected_names
  for items in zip(*data.values()):assert len({x['name']for x in items})==1 and len({x['device_characteristics_sha256']for x in items})==1
  sums={k:sum(x['bias_SSE_relative_to_requested_mean']for x in v)for k,v in data.items()};assert sums['ideal']==0 and sums['oracle']<sums['uncalibrated'];paired.append({'seed':seed,'bias_SSE':sums})
 assert r['cases'][0]['point_sha256']==json.loads((OUT/'gaussian_control_v1/BF16/parameters.json').read_text())['point_sha256']
 stats={};baseline=vals['baseline'][0]
 for kind in ['ideal','uncalibrated','oracle']:
  delta=np.array(vals[kind])-baseline;assert len(delta)==5;half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
  stats[kind]={'PPL_ratio':float(np.exp(delta.mean())),'MC95_t4':np.exp([delta.mean()-half,delta.mean()+half]).tolist()}
  for key in stats[kind]:np.testing.assert_allclose(stats[kind][key],r['statistics'][kind][key],rtol=1e-14)
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2);checks=[];seed=p['seeds'][0]
 ideal=json.loads((root/f'ideal_seed{seed}/parameters.json').read_text())['records'];order={x['name']:i for i,x in enumerate(ideal)}
 with safe_open(str(CHECKPOINT/'model.safetensors'),framework='pt',device='cpu')as source:
  for name in ['transformer.h.0.ln_1.weight','transformer.h.0.attn.c_proj.weight','transformer.ln_f.bias']:
   raw=source.get_tensor(name.removeprefix('transformer.')).to(torch.bfloat16).float().numpy();layout=p['preferences'].get(name,'storage_flat');program=FinitePBitProgram.encode(raw,group_size=128,layout=layout)
   for kind in ['ideal','uncalibrated','oracle']:
    expected=json.loads((root/f'{kind}_seed{seed}/parameters.json').read_text())['records'][order[name]]
    rngd=np.random.default_rng(np.random.SeedSequence([seed,order[name],11]));rngs=np.random.default_rng(np.random.SeedSequence([seed,order[name],22]));c_hash=hashlib.sha256();d_hash=hashlib.sha256();out=np.empty(raw.size,np.float32)
    for start,stop,lo,scale,_ in program._chunks():
     command=program.codes[start:stop].astype(np.float32)*np.float32(8/4095)-np.float32(4);target=2*command.astype(np.float64)
     gain=rngd.normal(1,.02,stop-start);offset=rngd.normal(0,.02,stop-start);d_hash.update(gain.tobytes());d_hash.update(offset.tobytes())
     if kind=='ideal':logit=target
     elif kind=='uncalibrated':logit=gain*target+offset
     else:
      request=(target-offset)/gain/2;code=np.rint(np.clip((request+4)*(4095/8),0,4095)).astype(np.uint16)
      field=code.astype(np.float32)*np.float32(8/4095)-np.float32(4);logit=2*gain*field.astype(np.float64)+offset
     prob=1/(1+np.exp(-logit));count=rngs.binomial(32768,prob).astype(np.uint16);c_hash.update(count.tobytes());out[start:stop]=lo+scale*count.astype(np.float64)/32768
    value=out.reshape(raw.shape[::-1]if layout=='transposed_matrix'else raw.shape);value=value.T.copy()if layout=='transposed_matrix'else value
    bf16=torch.from_numpy(value.copy()).to(torch.bfloat16)
    assert c_hash.hexdigest()==expected['count_sha256']and d_hash.hexdigest()==expected['device_characteristics_sha256']
    assert hashlib.sha256(bf16.view(torch.uint16).numpy().tobytes()).hexdigest()==expected['BF16_sha256'];checks.append({'name':name,'kind':kind})
 output={'status':'all16_scores_matched_devices_and_nine_weight_replays_audited','result_sha256':sha(root/'results.json'),'source_sha256':sha(__file__),'SDK_source_sha256':sha('publish/pbit1/src/pbit1/program.py'),'statistics':stats,'paired_biases':paired,'checks':checks,'scope':p['scope']}
 (OUT/'gpt2_calibration_control_v1_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(stats))
if __name__=='__main__':main()
