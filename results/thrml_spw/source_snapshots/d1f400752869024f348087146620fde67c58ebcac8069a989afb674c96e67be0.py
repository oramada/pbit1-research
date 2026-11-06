"""Five independent conventional CNN fits, validation selection, no test scoring."""
from pathlib import Path
import hashlib,json,datetime,time,copy,os,platform
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from results.thrml_spw.fashion_cnn_native.model import FashionCNN,features

OUT=Path('results/thrml_spw/fashion_cnn_native/training')
SRC=OUT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(state):return hashlib.sha256(b''.join(state[k].detach().cpu().numpy().tobytes() for k in sorted(state))).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def savedata(p,**a):
    q=p.with_suffix('.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**a)
    q.replace(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'No duplicate training'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    sources={str(f):sha(f) for f in sorted(SRC.glob('*.py'))}
    rawdir=Path('data/torchvision/FashionMNIST/raw');rawhash={str(f):sha(f) for f in sorted(rawdir.glob('*-ubyte'))};assert len(rawhash)==4
    p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources_sha256':sources,'dataset_raw_sha256':rawhash,'dataset':'FashionMNIST, existing standard splits; test data was consumed by previous studies','split_seed':20261029,'train_images':55000,'validation_images':5000,'test_images':10000,'architecture':'14x14 input; Conv1x8 k3 pad1, ReLU/maxpool2; Conv8x16 k3 pad1, ReLU/maxpool2; linear144x10; deterministic biases','weights':2664,'biases':34,'preprocessing':'2x2 average pool original28x28 grayscale /255 to14x14. No fitted normalization or augmentation.','training_device':'cpu','training_threads':4,'training_seeds':[0,1,2,3,4],'epochs':12,'batch_size':256,'optimizer':'Adam lr0.003, cosine schedule to0.0003 over12 epochs, no weight decay','selection':'Lowest full validation CE among12 fixed epochs, all5 model seeds retained; no test scoring in this program','initialization':'Standard independent signed PyTorch initialization reset separately by manual_seed(seed) before construction; no common checkpoint reload','future_native_encoding':'One normalized probability p=(w-c+r)/(2r) per signed weight; per-output-channel minmax center/radius. Free finite8-bit THRML fields±4; biases deterministic. This differs from literal positive w=p and metadata is counted.','future_activation_ranges':'Input[0,1]; hidden afterReLU/maxpool[0,1.25*max(selected-model full validation activations)] per layer/model. Deterministic quantized8-bit activations, exact padding masks.','future_plan':'All5models/10MCseeds, three predeclared T64/256/1024; freeze native inference, conventional controls and statistical protocols before test scoring. Training alone is not native inference or a full CNN result.','environment':{'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'CPU_deterministic':True,'CUDA_used':False}}
    write(OUT/'protocol.json',p)
    train=FashionMNIST('data/torchvision',train=True,download=False)
    assert len(train)==60000
    x=features(train.data);y=train.targets;ids=np.random.default_rng(p['split_seed']).permutation(60000);tr,va=ids[:55000],ids[55000:]
    savedata(OUT/'split.npz',train_indices=tr,validation_indices=va)
    savedata(OUT/'validation.npz',inputs=x[va].numpy(),labels=y[va].numpy())
    state={'complete':False,'protocol_sha256':sha(OUT/'protocol.json'),'rows':[],'test_scored':False};write(OUT/'manifest.json',state);started=time.monotonic();initials=[]
    for seed in p['training_seeds']:
        for f,h in sources.items():assert sha(f)==h
        torch.manual_seed(seed);model=FashionCNN();initial=digest(model.state_dict());assert initial not in initials;initials.append(initial)
        assert sum(v.numel() for k,v in model.state_dict().items() if k.endswith('weight'))==2664
        generator=torch.Generator().manual_seed(1507520+seed);opt=torch.optim.Adam(model.parameters(),lr=.003);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12,eta_min=.0003)
        history=[];bestce=np.inf;best=None;bestepoch=None
        for epoch in range(12):
            model.train();order=torch.randperm(len(tr),generator=generator);idx=torch.as_tensor(tr)[order];total=0.;steps=0
            for first in range(0,len(idx),256):
                ii=idx[first:first+256];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ii]),y[ii]);loss.backward();opt.step();total+=float(loss.detach())*len(ii);steps+=1
            model.eval();valce=0.;correct=0
            with torch.no_grad():
                for first in range(0,len(va),256):
                    ii=va[first:first+256];z=model(x[ii]);valce+=float(F.cross_entropy(z,y[ii],reduction='sum'));correct+=int((z.argmax(1)==y[ii]).sum())
            valce/=len(va)
            row={'epoch':epoch+1,'seed':seed,'train_ce':total/len(tr),'validation_ce':valce,'validation_accuracy':correct/len(va),'optimizer_steps':steps,'permutation_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest(),'epoch_model_sha256':digest(model.state_dict())}
            history.append(row)
            if valce<bestce:bestce=valce;best=copy.deepcopy(model.state_dict());bestepoch=epoch+1
            sched.step();write(OUT/f'seed{seed}_history.json',history);print(json.dumps({**row,'elapsed_seconds':time.monotonic()-started}),flush=True)
        model.load_state_dict(best);model.eval();logits=[];maxima=np.zeros(2)
        with torch.no_grad():
            for first in range(0,len(va),256):
                xb=x[va[first:first+256]];aa=model.activations(xb);maxima=np.maximum(maxima,[float(a.max()) for a in aa]);logits.append(model(xb).numpy())
        assert np.all(maxima>0)
        arrays={k.replace('.','_'):v.numpy() for k,v in best.items()};arrays.update(reference_validation_logits=np.concatenate(logits),activation_lower=np.array([0.,0.,0.]),activation_span=np.array([1.,1.25*maxima[0],1.25*maxima[1]]))
        path=OUT/f'seed{seed}.npz';savedata(path,**arrays);ckpt=OUT/f'seed{seed}.pt';tmp=ckpt.with_suffix('.tmp');torch.save(best,tmp);tmp.replace(ckpt)
        state['rows'].append({'seed':seed,'selected_epoch':bestepoch,'selected_validation_ce':bestce,'initial_state_sha256':initial,'selected_state_sha256':digest(best),'path':str(path),'sha256':sha(path),'checkpoint_path':str(ckpt),'checkpoint_sha256':sha(ckpt),'history_sha256':sha(OUT/f'seed{seed}_history.json'),'activation_ranges_from_validation_only':True,'test_scored':False});write(OUT/'manifest.json',state)
    for f,h in sources.items():assert sha(f)==h
    for f,h in rawhash.items():assert sha(f)==h
    assert len(state['rows'])==5 and len({r['selected_state_sha256'] for r in state['rows']})==5
    state.update(complete=True,elapsed_seconds=time.monotonic()-started);write(OUT/'manifest.json',state)
    print('Five conventional CNN checkpoints frozen; native inference and independent training audit pending',flush=True)
if __name__=='__main__':main()
