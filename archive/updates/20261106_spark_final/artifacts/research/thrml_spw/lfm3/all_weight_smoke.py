"""All-parameter LFM3 native-sampling integration, including tied embedding/head.

One short consumed prompt; this is a coverage/integration diagnostic, not a
benchmark or an assertion of BF16-equivalent quality or physical density.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'all_weight_smoke';out.mkdir(exist_ok=True)
    protocol={'source_sha256':sha(Path(__file__)),'worker_sha256':sha(Path(__file__).with_name('all_weight_worker.py')),
        'revision':'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7','samples':64,'seed':202610293,
        'stages':['identity_bridge','nominal_controls','native'],'input_sha256':sha(root/'baseline_smoke.npz'),
        'conversion':'Every unique learned weight, including all decoder matrices, convolution kernels, RMSNorm weights, and tied embedding/head. Tied weights counted once; each use sampled separately with identical programming law.',
        'range_policy':'Per output row for matrices and shared embedding/head; per entire vector for each norm. No per-coordinate range metadata.',
        'execution':'Signed ACT8 inputs, finite 8-bit fields [-4,4], int32 accumulation, FP64 affine readout cast to BF16 before original nonlinearities; original deterministic operators also execute before hooks, explicit extra simulation work.',
        'scope':'One consumed 16-token prompt and one seed. No benchmark/equivalence/physical-performance claim. Simulator banks time multiplex logical parameter coordinates.'}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',trust_remote_code=True,
        local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
    ids=torch.from_numpy(np.load(root/'baseline_smoke.npz')['input_ids']).to(device)
    code=sys.modules[type(model.model.layers[0].conv).__module__];code.can_use_lfm3_conv=lambda *args:False
    targets=[(n,m) for n,m in model.named_modules() if m._parameters.get('weight') is not None]
    unique={id(m.weight):m.weight for _,m in targets}
    total=sum(p.numel() for p in model.parameters());converted=sum(p.numel() for p in unique.values())
    assert converted==total==1728801536
    inventory={'unique_parameters':converted,'total_parameters':total,
        'modules':[{'name':n,'shape':list(m.weight.shape),
                    'shared_coordinate_group':'token_embedding_head' if m.weight is model.model.embed_tokens.weight else n,
                    'kind':'embedding' if isinstance(m,torch.nn.Embedding) else 'conv' if isinstance(m,torch.nn.Conv1d) else 'linear' if isinstance(m,torch.nn.Linear) else 'elementwise'} for n,m in targets]}
    (out/'conversion_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    with torch.inference_mode():baseline=model(input_ids=ids,use_cache=False).logits.float().cpu().numpy()
    np.testing.assert_array_equal(baseline,np.load(root/'baseline_smoke.npz')['logits'])
    mode='identity_bridge';worker=None;handles=[];calls=[];stderr=(out/'worker.log').open('w')
    def make_hook(name,module):
        def hook(mod,args,original):
            if mode=='identity_bridge':return original
            extra={};restore=None
            if isinstance(module,torch.nn.Embedding):
                tokens=args[0].detach().cpu().numpy();rows,inverse=np.unique(tokens,return_inverse=True)
                selected=module.weight[torch.from_numpy(rows).to(device)].detach().float().cpu().numpy()
                lower=selected.min(1);span=np.ptp(selected,axis=1)
                w=selected.reshape(1,-1);xx=np.ones((1,1,w.size),np.float32);kind='embedding'
                extra={'lower':np.repeat(lower,selected.shape[1]),'span':np.repeat(span,selected.shape[1])}
                restore=(tokens.shape,selected.shape,inverse)
            elif isinstance(module,torch.nn.Conv1d):
                x=args[0].detach().float().cpu().numpy()
                windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(0,0),(6,0))),7,axis=2)
                xx=windows.transpose(0,2,3,1).reshape(-1,7,x.shape[1])
                w=module.weight.detach().float().cpu().numpy()[:,0,:].T;kind='conv'
            elif isinstance(module,torch.nn.Linear):
                x=args[0].detach().float().cpu().numpy();xx=x.reshape(-1,x.shape[-1])
                w=module.weight.detach().float().cpu().numpy().T;kind='linear'
            else:
                assert module.weight.ndim==1 and hasattr(module,'variance_epsilon'),name
                z=args[0];zf=z.float()
                normalized=(zf*torch.rsqrt(zf.square().mean(-1,keepdim=True)+module.variance_epsilon)).to(z.dtype)
                x=normalized.detach().float().cpu().numpy();xx=x.reshape(-1,1,x.shape[-1])
                w=module.weight.detach().float().cpu().numpy().reshape(1,-1);kind='elementwise'
            ip=out/'runtime_input.npz';op=out/'runtime_output.npy';np.savez(ip,x=xx,w=w,**extra)
            req={'input':str(ip),'output':str(op),'kind':kind,'samples':64,'key':202610293,'call':len(calls),'mode':'native' if mode=='native' else 'mean'}
            worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush();line=worker.stdout.readline()
            if not line:raise RuntimeError('THRML worker terminated; preserve worker.log')
            receipt=json.loads(line);assert receipt['ok'] and receipt['finite']
            y=np.load(op)
            if kind=='embedding':
                shape,selected_shape,inverse=restore;y=y.reshape(selected_shape)[inverse].reshape(*shape,selected_shape[-1])
            elif kind=='conv':
                y=np.pad(y.reshape(x.shape[0],x.shape[2],x.shape[1]).transpose(0,2,1),((0,0),(0,0),(0,6)))
            else:y=y.reshape(*x.shape[:-1],w.shape[1])
            calls.append({'module':name,'stage':mode,'kind':kind,
                          'embedding_rows':rows.tolist() if kind=='embedding' else None,**receipt})
            (out/'progress.json').write_text(json.dumps({'stage':mode,'operator_calls':len(calls),'last_module':name})+'\n')
            return torch.from_numpy(y.copy()).to(device=original.device,dtype=original.dtype)
        return hook
    for n,m in targets:handles.append(m.register_forward_hook(make_hook(n,m)))
    raw={'input_ids':ids.cpu().numpy(),'BF16':baseline};metrics=[];begin=time.monotonic()
    try:
        for stage in protocol['stages']:
            mode=stage
            if stage=='nominal_controls':
                worker=subprocess.Popen([sys.executable,'-m','research.thrml_spw.gpu_environment','--',
                    '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.lfm3.all_weight_worker'],stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
                assert json.loads(worker.stdout.readline())['ready']
            with torch.inference_mode():logits=model(input_ids=ids,use_cache=False).logits.float().cpu().numpy()
            assert np.isfinite(logits).all()
            if stage=='identity_bridge':np.testing.assert_array_equal(logits,baseline)
            raw[stage]=logits;z=logits[:,:-1].astype(np.float64);mx=z.max(-1)
            ce=float(np.mean(mx+np.log(np.exp(z-mx[...,None]).sum(-1))-np.take_along_axis(z,ids[:,1:].cpu().numpy()[...,None],-1)[...,0]))
            metric={'stage':stage,'diagnostic_CE_FP64':ce,'logit_MSE_vs_BF16':float(np.mean((logits.astype(np.float64)-baseline)**2)),
                    'argmax_agreement':float(np.mean(logits.argmax(-1)==baseline.argmax(-1)))}
            metrics.append(metric);print(json.dumps(metric),flush=True)
    finally:
        for h in handles:h.remove()
        if worker is not None:
            if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.stdin.close()
            worker.wait(timeout=60)
        stderr.close()
    np.savez_compressed(out/'outputs.npz',**raw)
    (out/'results.json').write_text(json.dumps({'metrics':metrics,'operator_calls':calls,'unique_parameters_converted':converted,
        'total_parameters':total,'elapsed_seconds':time.monotonic()-begin,'outputs_sha256':sha(out/'outputs.npz'),
        'scope':protocol['scope']},indent=2)+'\n')


if __name__=='__main__':main()
