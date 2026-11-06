"""Real embedding/norm programming checks before full-coverage task execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from safetensors import safe_open


def main():
    out=Path('results/thrml_spw/lfm3_1p7b/all_weight_preflight');out.mkdir(exist_ok=True)
    protocol={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'worker_sha256':hashlib.sha256(Path(__file__).with_name('all_weight_worker.py').read_bytes()).hexdigest(),
        'samples':[1,64],'operators':['embedding','elementwise'],'modes':['mean','native'],
        'scope':'Programming identity and public-state exact replay for new operator types, no language task scoring.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    with safe_open('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors',framework='pt') as f:
        embed=f.get_tensor('model.embed_tokens.weight')[torch.tensor([124894,100])].float().numpy()
        norm=f.get_tensor('model.layers.0.operator_norm.weight').float().numpy()
    log=(out/'worker.log').open('w')
    worker=subprocess.Popen([sys.executable,'-m','research.thrml_spw.gpu_environment','--',
        '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.lfm3.all_weight_worker'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    assert json.loads(worker.stdout.readline())['ready'];receipts=[];saved={}
    try:
        for kind in ['embedding','elementwise']:
            if kind=='embedding':
                w=embed.reshape(1,-1);x=np.ones((1,1,w.size),np.float32)
                lo=np.repeat(embed.min(1),embed.shape[1]);sp=np.repeat(np.ptp(embed,axis=1),embed.shape[1])
                extra={'lower':lo,'span':sp}
            else:
                w=norm.reshape(1,-1);x=np.linspace(-1,1,2*w.size,dtype=np.float32).reshape(2,1,-1)
                lo=np.full(w.size,w.min(),np.float64);sp=np.full(w.size,float(w.max())-float(w.min()),np.float64);extra={}
            # Independent nominal controller mean, using shared row ranges for the embedding/head.
            p=np.divide(w-lo,sp,out=np.full_like(w,.5,dtype=np.float64),where=sp!=0).clip(1e-12,1-1e-12)
            f=np.clip((np.log(p)-np.log1p(-p))/2,-4,4)
            code=np.rint((f+4)*255/8).clip(0,255).astype(np.uint8)
            fields=code.astype(np.float32)*np.float32(8/255)-np.float32(4)
            probability=np.exp(-np.logaddexp(0,-2*fields.astype(np.float64)))
            scale=float(np.abs(x).max())/127
            quant=np.rint(x/scale).clip(-127,127).astype(np.int32)
            expected=(quant[:,0,:].astype(np.float64)*(lo+sp*probability))*scale
            for t in [1,64]:
                for mode in ['mean','native']:
                    ip=out/'runtime.npz';op=out/'runtime.npy';np.savez(ip,x=x,w=w,**extra)
                    req={'input':str(ip),'output':str(op),'kind':kind,'samples':t,'key':202610294,'call':len(receipts),'mode':mode,'replay':mode=='native'}
                    worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush();line=worker.stdout.readline()
                    if not line:raise RuntimeError('Worker terminated; inspect log')
                    receipt=json.loads(line);assert receipt['ok'] and receipt['finite']
                    y=np.load(op)
                    if mode=='mean':np.testing.assert_allclose(y,expected,rtol=2e-12,atol=2e-12)
                    else:assert receipt['public_replay_checked']
                    name=f'{kind}_{mode}_T{t}';saved[name]=y;receipts.append({'operator':kind,'mode':mode,'samples':t,**receipt})
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.stdin.close()
        worker.wait(timeout=60);log.close()
    np.savez_compressed(out/'outputs.npz',**saved)
    (out/'results.json').write_text(json.dumps({'status':'all_8_new_operator_checks_passed','receipts':receipts,
        'outputs_sha256':hashlib.sha256((out/'outputs.npz').read_bytes()).hexdigest()},indent=2)+'\n')
    print('All 8 embedding/norm checks passed',flush=True)


if __name__=='__main__':main()
