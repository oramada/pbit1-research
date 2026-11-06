"""Check SDK reconstruction against the frozen research sampler and real LFM weights."""
import json,sys
from pathlib import Path
import numpy as np
from .prepare_v1 import OUT,sha
from .counter_sampler_v3 import sample_parameter
from .lfm3_full_preflight_v1 import ROOT
sys.path.insert(0,str(Path('publish/pbit1/src').resolve()))
from pbit1 import FinitePBitProgram

def main():
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2)
 fixtures=[('boundary7',torch.linspace(-2,3,262563).reshape(37509,1,7).to(torch.bfloat16),'storage_flat',7,131072),
           ('transpose',torch.linspace(-2,3,323).reshape(17,19).to(torch.bfloat16),'transposed_matrix',128,32768)]
 policies=json.loads((OUT/'lfm3_calibration_v2/results.json').read_text())['policies']
 with safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as f:
  for name in ['model.layers.0.conv.conv.weight','model.layers.2.self_attn.q_norm.weight','model.layers.0.conv.out_proj.weight']:
   p=next(p for p in policies if p['name']==name);fixtures.append((name,f.get_tensor(name),p['selected_layout'],p['selected_group'],32768))
 results=[]
 for index,(name,raw,layout,group,T)in enumerate(fixtures):
  ref,record=sample_parameter(raw,index,layout,group,T,torch,2026112001)
  program=FinitePBitProgram.encode(raw.float().numpy(),group_size=group,layout=layout)
  got=program.reconstruct(T,seed=2026112001,parameter_index=index)
  assert torch.equal(torch.from_numpy(got.weights).to(torch.bfloat16),ref)
  assert got.counts_sha256==record['counts_sha256']and got.maximum_count==record['max_count']and got.counter_dtype==record['counter_dtype']
  results.append({'name':name,'parameters':raw.numel(),'group':group,'layout':layout,'samples':T,'counts_sha256':got.counts_sha256,'BF16_equal':True})
 output={'status':'all5_sdk_reconstructions_bitwise_match_frozen_sampler','source_sha256':sha(__file__),'SDK_source_sha256':sha('publish/pbit1/src/pbit1/program.py'),'research_sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v3.py')),'fixtures':results}
 (OUT/'sdk_program_verification_v2.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
if __name__=='__main__':main()
