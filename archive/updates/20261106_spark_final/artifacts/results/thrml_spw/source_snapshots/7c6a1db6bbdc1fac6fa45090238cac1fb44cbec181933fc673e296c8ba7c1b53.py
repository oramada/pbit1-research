"""Full first-layer operator exports; no task scoring or model conversion."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def main():
    torch.set_num_threads(4)
    root=Path('results/thrml_spw/lfm3_1p7b')
    pilot=np.load(root/'operator_pilot_export.npz')
    x=torch.from_numpy(pilot['linear_inputs'])
    with safe_open('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors',framework='pt') as f:
        w=f.get_tensor('model.layers.0.conv.in_proj.weight').float()
        conv=f.get_tensor('model.layers.0.conv.conv.weight').float()[:,0,:]
    bc=F.linear(x,w).T.unsqueeze(0)
    b,c,value=bc.chunk(3,dim=1)
    windows=F.pad(b*value,(6,0)).unfold(-1,7,1).permute(0,2,3,1).reshape(-1,7,2048)
    target=root/'full_operator_export.npz'
    np.savez_compressed(target,linear_inputs=x.numpy(),linear_weight=w.T.contiguous().numpy(),
                        conv_inputs=windows.numpy(),conv_weight=conv.T.contiguous().numpy())
    info={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'export_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'linear_shape':[2048,6144],'conv_shape':[7,2048],
          'scope':'Full first-layer input projection and depthwise convolution kernels; 16-token FP32 reference activations. Independent operator tests, not a composed stochastic layer.'}
    (root/'full_operator_export.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
