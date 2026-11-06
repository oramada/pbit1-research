"""Check exported CNNs, splits, selection, all validation outputs and ranges."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from results.thrml_spw.fashion_cnn_native.model import FashionCNN,features
OUT=Path('results/thrml_spw/fashion_cnn_native/training')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(b''.join(s[k].detach().cpu().numpy().tobytes() for k in sorted(s))).hexdigest()
def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and not m['test_scored'] and m['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in {**p['sources_sha256'],**p['dataset_raw_sha256']}.items():assert sha(f)==h
    data=FashionMNIST('data/torchvision',train=True,download=False);split=np.random.default_rng(20261029).permutation(60000);s=np.load(OUT/'split.npz');np.testing.assert_array_equal(s['train_indices'],split[:55000]);np.testing.assert_array_equal(s['validation_indices'],split[55000:]);assert not np.intersect1d(s['train_indices'],s['validation_indices']).size
    # Independently form2x2 means from raw images, rather than use model features().
    raw=data.data[split[55000:]].numpy().astype(np.float32)/np.float32(255)
    x=((raw[:,0::2,0::2]+raw[:,0::2,1::2]+raw[:,1::2,0::2]+raw[:,1::2,1::2])/np.float32(4))[:,None]
    v=np.load(OUT/'validation.npz');np.testing.assert_array_equal(x,v['inputs']);np.testing.assert_array_equal(data.targets[split[55000:]].numpy(),v['labels']);xx=torch.from_numpy(x);rows=[];initials=[]
    for row in m['rows']:
        seed=row['seed'];torch.manual_seed(seed);model=FashionCNN();initial=digest(model.state_dict());assert initial==row['initial_state_sha256'] and initial not in initials;initials.append(initial)
        assert sha(row['path'])==row['sha256'] and sha(row['checkpoint_path'])==row['checkpoint_sha256'];historypath=OUT/f'seed{seed}_history.json';assert sha(historypath)==row['history_sha256'];history=json.loads(historypath.read_text());assert len(history)==12
        best=min(history,key=lambda h:h['validation_ce']);assert best['epoch']==row['selected_epoch'] and best['validation_ce']==row['selected_validation_ce'] and best['epoch_model_sha256']==row['selected_state_sha256']
        generator=torch.Generator().manual_seed(1507520+seed)
        for epoch,h in enumerate(history,1):
            order=torch.randperm(55000,generator=generator);assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==h['permutation_sha256'] and h['optimizer_steps']==215 and h['epoch']==epoch
        state=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=True);assert digest(state)==row['selected_state_sha256'];a=np.load(row['path'])
        for k,val in state.items():np.testing.assert_array_equal(val.numpy(),a[k.replace('.','_')])
        model.load_state_dict(state);model.eval();refs=[];maxima=np.zeros(2);error=0.
        with torch.no_grad():
            for first in range(0,5000,256):
                xb=xx[first:first+256]
                first_layer=F.max_pool2d(F.relu(F.conv2d(xb,torch.from_numpy(a['conv1_weight']),torch.from_numpy(a['conv1_bias']),padding=1)),2)
                second_layer=F.max_pool2d(F.relu(F.conv2d(first_layer,torch.from_numpy(a['conv2_weight']),torch.from_numpy(a['conv2_bias']),padding=1)),2)
                z=F.linear(second_layer.flatten(1),torch.from_numpy(a['head_weight']),torch.from_numpy(a['head_bias']));np.testing.assert_array_equal(z.numpy(),model(xb).numpy());refs.append(z.numpy());maxima=np.maximum(maxima,[float(first_layer.max()),float(second_layer.max())])
        ref=np.concatenate(refs);np.testing.assert_array_equal(ref,a['reference_validation_logits']);np.testing.assert_array_equal(a['activation_lower'],[0,0,0]);np.testing.assert_array_equal(a['activation_span'],[1,1.25*maxima[0],1.25*maxima[1]])
        # Reproduce the originally batched reduction, not a differently associated sum.
        ce=sum(float(F.cross_entropy(torch.from_numpy(ref[i:i+256]),torch.from_numpy(v['labels'][i:i+256]),reduction='sum')) for i in range(0,5000,256))/5000;assert ce==row['selected_validation_ce']
        rows.append({'seed':seed,'selected_epoch':row['selected_epoch'],'validation_accuracy':float(np.mean(ref.argmax(1)==v['labels'])),'validation_ce':ce,'all50000_validation_logits_exact':True,'ranges_validation_only':True})
    assert len(rows)==5
    audit={'passed':True,'scope':'Conventional checkpoints/export/split/selection/initialization/minibatch schedules/full validation outputs and ranges. Not independent full retraining, native inference, fresh test confirmation or hardware evidence.','models':rows,'source_sha256':p['sources_sha256'],'test_scored':False}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
