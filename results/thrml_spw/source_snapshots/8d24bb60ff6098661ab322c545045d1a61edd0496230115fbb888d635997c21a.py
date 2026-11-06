"""Separate conventional floating-format controls; no p-bit sampling."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from results.thrml_spw.fashion_cnn_native.model import FashionCNN
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'precision_evidence';PARENT=ROOT/'cohort_evidence';TRAIN=ROOT/'training'
NAMES=['FP16_weights_F64','BF16_weights_F64','FP16_Torch_CPU','BF16_Torch_CPU']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def save(p,**a):
    q=p.with_suffix('.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**a)
    q.replace(p)
def guard(p):
    for f,h in {**p['source_sha256'],**p['input_sha256']}.items():assert sha(f)==h,f
def parent_controls(p):
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256'];c=json.loads((PARENT/'controls.json').read_text());assert c['complete'] and c['protocol_sha256']==p['parent_protocol_sha256'] and c['test_data_sha256']==sha(PARENT/'test_data.npz')
def parent_closed(p):
    parent_controls(p);assert json.loads((PARENT/'pipeline.json').read_text())['status']=='full_native_CNN_independently_audited_goal_active'
    for f,h in json.loads((PARENT/'completion.json').read_text())['artifact_sha256'].items():assert sha(f)==h,f
    assert json.loads((PARENT/'native_audit.json').read_text())['passed'] and json.loads((PARENT/'statistics_audit.json').read_text())['passed']
def output(x,row,name):
    dtype=torch.float16 if name.startswith('FP16') else torch.bfloat16
    if name.endswith('Torch_CPU'):
        model=FashionCNN();model.load_state_dict(torch.load(row['checkpoint_path'],map_location='cpu',weights_only=True));model=model.to(dtype=dtype).eval()
        with torch.no_grad():z=model(torch.as_tensor(x,dtype=dtype))
        assert z.dtype==dtype;return z.float().numpy().astype(float)
    model=np.load(row['path']);h=torch.as_tensor(x,dtype=torch.float64)
    with torch.no_grad():
        for li,prefix in enumerate(['conv1','conv2','head']):
            w=torch.from_numpy(model[prefix+'_weight']).to(dtype).to(torch.float64);b=torch.as_tensor(model[prefix+'_bias'],dtype=torch.float64)
            h=F.max_pool2d(F.relu(F.conv2d(h,w,b,padding=1)),2) if li<2 else F.linear(h.flatten(1),w,b)
    return h.numpy()
def metric(z,y):
    top=z.max(1);loss=top+np.log(np.exp(z-top[:,None]).sum(1))-z[np.arange(len(y)),y];return np.array([np.mean(z.argmax(1)==y),loss.mean()])
