"""Conventional controls: selfcheck on validation, test only after source seal."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT,OUT,TRAIN,CONTROLS,sha,write,save,guard,parent_closed,control_weights,numpy_control
def torch_control(x,model,weights,name):
    dtype=torch.float32 if name=='FP32_F32' else torch.float64;h=torch.as_tensor(x,dtype=dtype)
    for li,w in enumerate(weights):
        if name.startswith('ACT8'):
            span=float(model['activation_span'][li]);h=span*torch.round(255*torch.clamp(h/span,0,1))/255
        ww=torch.as_tensor(w,dtype=dtype);b=torch.as_tensor(model[['conv1_bias','conv2_bias','head_bias'][li]],dtype=dtype)
        if li<2:h=F.max_pool2d(F.relu(F.conv2d(h,ww,b,padding=1)),2)
        else:h=F.linear(h.flatten(1),ww,b)
    return h.numpy().astype(float)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selfcheck','run']);args=parser.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);train=json.loads((TRAIN/'manifest.json').read_text())
    if args.action=='selfcheck':
        path=ROOT/'control_selfcheck.json';assert not path.exists();x=np.load(TRAIN/'validation.npz')['inputs'][:64];rows=[]
        for mi,row in enumerate(train['rows']):
            model=np.load(row['path'])
            for name in CONTROLS:
                w=control_weights(model,name);z=torch_control(x,model,w,name);reference=numpy_control(x,model,w,name.startswith('ACT8'));tol=(2e-5,1e-4) if name=='FP32_F32' else (2e-12,2e-11);np.testing.assert_allclose(z,reference,rtol=tol[0],atol=tol[1]);np.testing.assert_array_equal(z.argmax(1),reference.argmax(1));rows.append({'model':mi,'control':name,'max_error':float(np.max(abs(z-reference)))})
        write(path,{'passed':True,'scope':'50 validation-only numerical control paths,64inputs, no labels/scoring or test images; Torch versus NumPy spatial contractions','source_sha256':{str(f):sha(f) for f in [Path(__file__),ROOT/'cohort/common.py']},'rows':rows});print('Conventional CNN validation selfcheck passes50paths',flush=True);return
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();assert not (OUT/'controls.json').exists();test=FashionMNIST('data/torchvision',train=False,download=False);x=F.avg_pool2d(test.data[:,None].float()/255,2).numpy();labels=test.targets.numpy();save(OUT/'test_data.npz',inputs=x,labels=labels);arrays={};rows=[]
    with torch.no_grad():
        for mi,row in enumerate(train['rows']):
            model=np.load(row['path'])
            for ci,name in enumerate(CONTROLS):
                w=control_weights(model,name);parts=[torch_control(x[first:first+128],model,w,name) for first in range(0,10000,128)];z=np.concatenate(parts);assert z.shape==(10000,10) and np.isfinite(z).all();arrays[f'c{ci}_model{mi}']=z;rows.append({'control':name,'model':mi});print(json.dumps(rows[-1]),flush=True)
    save(OUT/'control_logits.npz',**arrays);guard(p);write(OUT/'controls.json',{'complete':True,'rows':rows,'test_data_sha256':sha(OUT/'test_data.npz'),'control_logits_sha256':sha(OUT/'control_logits.npz'),'protocol_sha256':sha(OUT/'protocol.json'),'scope':'All source-sealed controls retained; dequantized CPU arithmetic, no hardware integer-kernel/energy claim. Test data consumed historically; no fresh generalization confirmation.'})
if __name__=='__main__':main()
