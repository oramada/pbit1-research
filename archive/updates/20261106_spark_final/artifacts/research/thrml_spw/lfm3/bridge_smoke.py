"""End-to-end LFM3 forward bridge; explicit progressive conversion coverage.

This single consumed prompt is an integration diagnostic, not a language benchmark.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'bridge_smoke';out.mkdir(exist_ok=True)
    protocol={'model_revision':'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
              'source_sha256':sha(Path(__file__)),'worker_sha256':sha(Path(__file__).with_name('bridge_worker.py')),
              'samples':16,'seed':202610292,'stages':['identity_bridge','nominal_controls','native'],
              'conversion':'all decoder Linear projections and all depthwise convolutions; tied embedding/head and norm parameters deterministic',
              'inputs_sha256':sha(root/'baseline_smoke.npz'),
              'scope':'16-token previously consumed smoke prompt; next-token CE and logit comparisons are diagnostics only. No benchmark perplexity, equivalence or all-parameter conversion claim.',
              'runtime':'ACT8 per operator tensor; finite 8-bit fields [-4,4]; int32 sums before BF16 cast and original downstream nonlinearities. Original deterministic operators also execute before replacement hooks, charged as simulator overhead.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',trust_remote_code=True,
            local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
    ids=torch.from_numpy(np.load(root/'baseline_smoke.npz')['input_ids']).to(device)
    with torch.inference_mode():original_baseline=model(input_ids=ids,use_cache=False).logits.float().cpu().numpy()
    previous=np.load(root/'baseline_smoke.npz')['logits']
    np.testing.assert_array_equal(original_baseline,previous)
    code=sys.modules[type(model.model.layers[0].conv).__module__]
    # Force module-based width-7 convolution so hooks cannot be bypassed by the custom kernel.
    code.can_use_lfm3_conv=lambda *args:False
    targets=[(n,m) for n,m in model.named_modules() if n.startswith('model.layers.') and isinstance(m,(torch.nn.Linear,torch.nn.Conv1d))]
    converted=sum(m.weight.numel() for _,m in targets)
    protocol_inventory={'modules':[{'name':n,'shape':list(m.weight.shape)} for n,m in targets],
                        'converted_unique_parameters':converted,'total_unique_parameters':sum(p.numel() for p in model.parameters())}
    (out/'conversion_inventory.json').write_text(json.dumps(protocol_inventory,indent=2)+'\n')
    with torch.inference_mode():baseline=model(input_ids=ids,use_cache=False).logits.float().cpu().numpy()
    mode='identity_bridge';calls=[];worker=None;handles=[]
    stderr=(out/'worker.log').open('w')
    def hook(name,module):
        def replace(mod,args,original):
            nonlocal calls
            if mode=='identity_bridge':return original
            x=args[0].detach().float().cpu().numpy()
            if isinstance(module,torch.nn.Conv1d):
                assert module.groups==module.in_channels and module.kernel_size==(7,)
                windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(0,0),(6,0))),7,axis=2)
                xx=windows.transpose(0,2,3,1).reshape(-1,7,x.shape[1])
                w=module.weight.detach().float().cpu().numpy()[:,0,:].T
                kind='conv'
            else:
                xx=x.reshape(-1,x.shape[-1]);w=module.weight.detach().float().cpu().numpy().T;kind='linear'
            ip=out/'runtime_input.npz';op=out/'runtime_output.npy'
            np.savez(ip,x=xx,w=w)
            req={'input':str(ip),'output':str(op),'kind':kind,'samples':16,'key':202610292,'call':len(calls),'mode':'native' if mode=='native' else 'mean'}
            worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush()
            line=worker.stdout.readline()
            if not line:raise RuntimeError('Native worker terminated; inspect worker.log')
            receipt=json.loads(line);assert receipt['ok'] and receipt['finite']
            y=np.load(op)
            if kind=='conv':
                y=y.reshape(x.shape[0],x.shape[2],x.shape[1]).transpose(0,2,1)
                y=np.pad(y,((0,0),(0,0),(0,6)))
            else:y=y.reshape(*x.shape[:-1],w.shape[1])
            calls.append({'module':name,'stage':mode,**receipt})
            (out/'progress.json').write_text(json.dumps({'stage':mode,'operator_calls':len(calls),'last_module':name})+'\n')
            return torch.from_numpy(y.copy()).to(device=original.device,dtype=original.dtype)
        return replace
    for n,m in targets:handles.append(m.register_forward_hook(hook(n,m)))
    raw={'input_ids':ids.cpu().numpy(),'BF16':baseline,'original_kernel_BF16':original_baseline};metrics=[]
    start=time.monotonic()
    try:
        for stage in protocol['stages']:
            mode=stage
            if stage=='nominal_controls':
                worker=subprocess.Popen([sys.executable,'-m','research.thrml_spw.gpu_environment','--',
                    '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.lfm3.bridge_worker'],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
                ready=json.loads(worker.stdout.readline());assert ready['ready']
            with torch.inference_mode():logits=model(input_ids=ids,use_cache=False).logits.float().cpu().numpy()
            if stage=='identity_bridge':np.testing.assert_array_equal(logits,baseline)
            assert np.isfinite(logits).all()
            raw[stage]=logits
            ce=F.cross_entropy(torch.from_numpy(logits[:,:-1].reshape(-1,logits.shape[-1])),ids[:,1:].cpu().reshape(-1)).item()
            metrics.append({'stage':stage,'diagnostic_next_token_CE':ce,
                'logit_MSE_vs_BF16':float(np.mean((logits.astype(np.float64)-baseline)**2)),
                'argmax_agreement':float(np.mean(logits.argmax(-1)==baseline.argmax(-1)))})
            print(json.dumps(metrics[-1]),flush=True)
    finally:
        for h in handles:h.remove()
        if worker is not None:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.stdin.close()
            worker.wait(timeout=60)
        stderr.close()
    np.savez_compressed(out/'outputs.npz',**raw)
    (out/'results.json').write_text(json.dumps({'metrics':metrics,'operator_calls':calls,
        'converted_unique_parameters':converted,'total_unique_parameters':protocol_inventory['total_unique_parameters'],
        'elapsed_seconds':time.monotonic()-start,'outputs_sha256':sha(out/'outputs.npz'),
        'BF16_convolution_kernel_switch_logit_MSE':float(np.mean((baseline.astype(np.float64)-original_baseline)**2)),
        'scope':protocol['scope']},indent=2)+'\n')


if __name__=='__main__':main()
