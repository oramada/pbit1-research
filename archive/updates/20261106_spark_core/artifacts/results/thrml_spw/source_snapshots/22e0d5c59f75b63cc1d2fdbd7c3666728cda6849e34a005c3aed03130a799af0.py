"""Export real first-layer LFM3 weights/activations for bounded operator validation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def main():
    torch.set_num_threads(4)
    root = Path('results/thrml_spw/lfm3_1p7b')
    ids = np.load(root/'baseline_smoke.npz')['input_ids']
    cp = Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    with safe_open(cp, framework='pt') as f:
        embed = f.get_tensor('model.embed_tokens.weight')[torch.from_numpy(ids)].float()
        norm = f.get_tensor('model.layers.0.operator_norm.weight').float()
        w = f.get_tensor('model.layers.0.conv.in_proj.weight').float()
        conv = f.get_tensor('model.layers.0.conv.conv.weight').float()[:16,0,:]
    # FP32 reference activations, explicitly distinct from the BF16 full-model path.
    x = embed * torch.rsqrt(embed.square().mean(-1,keepdim=True)+1e-5) * norm
    bc = F.linear(x, w).transpose(-1,-2)
    b, c, value = bc.chunk(3,dim=-2)
    u = (b*value)[:,:16,:]
    windows = F.pad(u,(6,0)).unfold(-1,7,1).permute(0,2,3,1).reshape(-1,7,16)
    x = x.reshape(-1,2048)
    arrays = {'linear_inputs':x.numpy(),'linear_weight':w[:16].T.contiguous().numpy(),
              'conv_inputs':windows.numpy(),'conv_weight':conv.T.contiguous().numpy()}
    arrays['linear_reference'] = arrays['linear_inputs'].astype(np.float64) @ arrays['linear_weight'].astype(np.float64)
    arrays['conv_reference'] = np.einsum('bki,ki->bi', arrays['conv_inputs'].astype(np.float64), arrays['conv_weight'].astype(np.float64))
    target=root/'operator_pilot_export.npz'
    np.savez_compressed(target,**arrays)
    (root/'operator_pilot_export.json').write_text(json.dumps({
        'checkpoint_revision':'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'export_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'linear_weights':32768,'convolution_weights':112,
        'scope':'First 16 output channels only; FP32 checkpoint-derived first-layer activations. No whole-model conversion or quality claim.'},indent=2)+'\n')
    print('Exported', {k:list(v.shape) for k,v in arrays.items()},flush=True)


if __name__=='__main__':main()
