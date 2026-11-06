"""Precision grid mean-law and native exact replay checks on preserved inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'precision_preflight';out.mkdir(exist_ok=True)
    inputs={k:root/'group128_preflight'/f'input_{i}.npz' for k,i in [('linear',0),('conv',4),('embedding',8),('elementwise',12)]}
    protocol={'source_sha256':sha(Path(__file__)),'worker_sha256':sha(Path(__file__).with_name('precision_worker.py')),
        'inputs_sha256':{k:sha(v) for k,v in inputs.items()},'field_bits':[8,12],'activation_bits':[8,12],
        'samples':64,'modes':['mean','native'],'scope':'Mean-law/encoding precision and native public trajectory validation; no language task scoring.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    log=(out/'worker.log').open('w')
    worker=subprocess.Popen([sys.executable,'-m','research.thrml_spw.gpu_environment','--',
        '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.lfm3.precision_worker'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    assert json.loads(worker.stdout.readline())['ready'];rows=[];saved={}
    try:
        for fb in [8,12]:
            for ab in [8,12]:
                for kind,path in inputs.items():
                    a=np.load(path);w=a['w'].astype(np.float64);x=a['x'];size=min(128,w.shape[0])
                    grouped=w.reshape(-1,size,w.shape[1]);lo=grouped.min(1)[:,None,:];span=np.ptp(grouped,axis=1)[:,None,:]
                    if kind in ['embedding','elementwise']:lo=a['lower'][None,None,:];span=a['span'][None,None,:]
                    p=np.divide(grouped-lo,span,out=np.full_like(grouped,.5),where=span!=0).clip(1e-12,1-1e-12)
                    field=((np.log(p)-np.log1p(-p))/2).clip(-4,4);levels=(1<<fb)-1
                    code=np.rint((field+4)*levels/8).clip(0,levels).astype(np.uint16)
                    h=code.astype(np.float32)*np.float32(8/levels)-np.float32(4)
                    q=np.exp(-np.logaddexp(0,-2*h.astype(np.float64)));wm=(lo+span*q).reshape(w.shape)
                    maximum=(1<<(ab-1))-1;scale=float(np.abs(x).max())/maximum if np.any(x) else 1.0
                    quant=np.rint(x/scale).clip(-maximum,maximum).astype(np.int32).astype(np.float64)*scale
                    expected=quant@wm if kind=='linear' else np.sum(quant*wm[None,:,:],axis=1)
                    for mode in ['mean','native']:
                        call=len(rows);op=out/f'output_{call}.npy'
                        req={'input':str(path),'output':str(op),'kind':kind,'samples':64,'field_bits':fb,'activation_bits':ab,
                            'key':202610297,'call':call,'mode':mode,'replay':mode=='native','state_dir':str(out/'raw_states')}
                        worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush();line=worker.stdout.readline()
                        if not line:raise RuntimeError('Worker terminated; preserve logs')
                        receipt=json.loads(line);assert receipt['ok'] and receipt['finite']
                        assert receipt['field_bits']==fb and receipt['activation_bits']==ab
                        y=np.load(op)
                        if mode=='mean':np.testing.assert_allclose(y,expected,rtol=2e-12,atol=2e-12)
                        else:assert receipt['public_replay_checked']
                        rows.append({'kind':kind,'mode':mode,'input':str(path),**receipt})
                        saved[f'f{fb}_a{ab}_{kind}_{mode}']=y
                print(f'field{fb}/ACT{ab} passed',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.stdin.close()
        worker.wait(timeout=60);log.close()
    np.savez_compressed(out/'outputs.npz',**saved)
    artifacts=[*sorted((out/'raw_states').glob('*.npz')),*sorted(out.glob('output_*.npy')),out/'outputs.npz']
    (out/'results.json').write_text(json.dumps({'status':'all32_precision_checks_passed','receipts':rows,
        'artifacts_sha256':{str(p.relative_to(out)):sha(p) for p in artifacts}},indent=2)+'\n')


if __name__=='__main__':main()
