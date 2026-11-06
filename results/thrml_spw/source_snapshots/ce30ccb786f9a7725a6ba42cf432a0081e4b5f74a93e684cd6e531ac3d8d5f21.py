"""Actual LFM3 slices: grouped means, tied controls, exact public-state replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from safetensors import safe_open


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def encode(grouped,low,span):
    p=np.divide(grouped-low,span,out=np.full_like(grouped,.5,dtype=np.float64),where=span!=0).clip(1e-12,1-1e-12)
    h=((np.log(p)-np.log1p(-p))/2).clip(-4,4)
    c=np.rint((h+4)*255/8).clip(0,255).astype(np.uint8)
    f=c.astype(np.float32)*np.float32(8/255)-np.float32(4)
    q=np.exp(-np.logaddexp(0,-2*f.astype(np.float64)))
    return c,low+span*q


def main():
    torch.set_num_threads(4)
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'group128_preflight';out.mkdir(exist_ok=True)
    protocol={'source_sha256':sha(Path(__file__)),'worker_sha256':sha(Path(__file__).with_name('group128_worker.py')),
        'operators':['linear','conv','embedding','elementwise'],'samples':[1,64],'modes':['mean','native'],
        'scope':'Grouped mean/readout and exact integer replay checks; actual LFM3 slices, padded final tiles, embeddings/head shared control identity; no language scores.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    full=np.load(root/'full_operator_export.npz')
    with safe_open('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors',framework='pt') as f:
        embed=f.get_tensor('model.embed_tokens.weight')[torch.tensor([124894,100])].float().numpy()
        norm=f.get_tensor('model.layers.0.operator_norm.weight').float().numpy()
    stderr=(out/'worker.log').open('w')
    worker=subprocess.Popen([sys.executable,'-m','research.thrml_spw.gpu_environment','--',
        '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.lfm3.group128_worker'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
    assert json.loads(worker.stdout.readline())['ready'];receipts=[];saved={};tied=False
    try:
        for kind in protocol['operators']:
            extra={}
            if kind=='linear':w=full['linear_weight'][:256,:70];x=full['linear_inputs'][:,:256]
            elif kind=='conv':w=full['conv_weight'][:,:70];x=full['conv_inputs'][:,:,:70]
            else:
                original=embed if kind=='embedding' else norm.reshape(1,-1)
                grouped=original.astype(np.float64).reshape(original.shape[0],-1,128)
                low=grouped.min(2);span=np.ptp(grouped,axis=2)
                extra={'lower':np.repeat(low.reshape(-1),128),'span':np.repeat(span.reshape(-1),128)}
                w=original.reshape(1,-1)
                x=np.ones((1,1,w.size),np.float32) if kind=='embedding' else np.linspace(-1,1,2*w.size,dtype=np.float32).reshape(2,1,-1)
                if kind=='embedding':
                    c,_=encode(grouped,low[:,:,None],span[:,:,None])
                    head=embed.T.astype(np.float64).reshape(16,128,2)
                    hc,_=encode(head,head.min(1)[:,None,:],np.ptp(head,axis=1)[:,None,:])
                    np.testing.assert_array_equal(c.transpose(1,2,0),hc);tied=True
            scale=float(np.max(np.abs(x)))/127 if np.any(x) else 1.0
            quant=np.rint(x/scale).clip(-127,127).astype(np.int32).astype(np.float64)*scale
            if kind in ['embedding','elementwise']:
                _,wm=encode(w.astype(np.float64),extra['lower'][None,:],extra['span'][None,:])
                expected=quant[:,0,:]*wm
            else:
                size=min(128,w.shape[0]);g=w.astype(np.float64).reshape(-1,size,w.shape[1])
                _,wm=encode(g,g.min(1)[:,None,:],np.ptp(g,axis=1)[:,None,:]);wm=wm.reshape(w.shape)
                expected=quant@wm if kind=='linear' else np.sum(quant*wm[None,:,:],axis=1)
            for t in protocol['samples']:
                for mode in protocol['modes']:
                    call=len(receipts);ip=out/f'input_{call}.npz';op=out/f'output_{call}.npy'
                    np.savez_compressed(ip,x=x,w=w,**extra)
                    req={'input':str(ip),'output':str(op),'kind':kind,'samples':t,'key':202610296,
                        'call':call,'mode':mode,'replay':mode=='native','state_dir':str(out/'raw_states')}
                    worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush();line=worker.stdout.readline()
                    if not line:raise RuntimeError('Worker terminated; inspect preserved logs')
                    receipt=json.loads(line);assert receipt['ok'] and receipt['finite']
                    y=np.load(op)
                    if mode=='mean':np.testing.assert_allclose(y,expected,rtol=2e-12,atol=2e-12)
                    else:assert receipt['public_replay_checked']
                    saved[f'{kind}_{mode}_T{t}']=y;receipts.append({'kind':kind,'mode':mode,'samples':t,**receipt})
            print(kind,'passed',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.stdin.close()
        worker.wait(timeout=60);stderr.close()
    assert tied
    np.savez_compressed(out/'outputs.npz',**saved)
    files=[*sorted((out/'raw_states').glob('*.npz')),*sorted(out.glob('input_*.npz')),*sorted(out.glob('output_*.npy')),out/'outputs.npz']
    (out/'results.json').write_text(json.dumps({'status':'all16_grouped_checks_passed','tied_embedding_head_codes_exact':tied,
        'receipts':receipts,'artifacts_sha256':{str(p.relative_to(out)):sha(p) for p in files}},indent=2)+'\n')


if __name__=='__main__':main()
