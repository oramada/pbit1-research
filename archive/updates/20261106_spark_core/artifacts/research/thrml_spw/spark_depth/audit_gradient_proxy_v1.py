"""Check allocation arithmetic and selected full-network gradients on CPU."""
from pathlib import Path
import json,math
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha

def main():
    source=OUT/'loss_gradient_variance_v1/results.json';data=json.loads(source.read_text());lookup={r['name']:r for r in data['parameters']}
    assert sum(r['elements']for r in lookup.values())==124439808
    for proposal in data['allocation_proposals']:
        value=0.;cost=0
        for p in proposal['parameters']:
            assert p['samples']in[2**j for j in range(10,18)]
            row=lookup[p['name']];S=row['transposed_S']if p['layout']=='transposed_matrix'else row['storage_S']
            assert S==p['proxy_coefficient']and row['elements']==p['parameters'];value+=S/p['samples'];cost+=p['parameters']*p['samples']
        assert np.isclose(value,proposal['proxy_variance'])and value<=proposal['target_proxy_variance']*(1+1e-12)
        assert cost==proposal['observations']and np.isclose(cost/(124439808*32768),proposal['versus_uniform_32768_cost'])
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval()
    with torch.no_grad():
        for p in model.parameters():p.copy_(p.to(torch.bfloat16).float())
    with np.load(OUT/'training_diagnostic_windows.npz')as f:ids=torch.from_numpy(f['input_ids'][:1,:129].copy())
    model.zero_grad(set_to_none=True);out=model(input_ids=ids,labels=ids,use_cache=False);out.loss.backward();params=dict(model.named_parameters());checks=[]
    for name in ['transformer.ln_f.bias','transformer.h.0.ln_1.weight','transformer.h.0.attn.c_proj.weight','transformer.wte.weight']:
        p=params[name];gradient=p.grad.detach().numpy();index=np.unravel_index(np.argmax(np.abs(gradient)),gradient.shape);g=float(gradient[index]);base=float(p[index].detach());estimates=[]
        try:
            for epsilon in [.003,.001]:
                with torch.no_grad():
                    p[index]=base+epsilon;plus=float(model(input_ids=ids,labels=ids,use_cache=False).loss)
                    p[index]=base-epsilon;minus=float(model(input_ids=ids,labels=ids,use_cache=False).loss)
                estimates.append((plus-minus)/(2*epsilon))
        finally:
            with torch.no_grad():p[index]=base
        errors=[abs(x-g)for x in estimates];passed=min(errors)<=.0003+.05*abs(g);assert passed,(name,g,estimates)
        checks.append({'parameter':name,'coordinate':[int(x)for x in index],'autograd':g,'finite_difference_derivatives':estimates,'passed':passed})
    assert not torch.cuda.is_initialized();dest=OUT/'gradient_proxy_audit_v1.json';dest.write_text(json.dumps({'status':'allocation_arithmetic_and_four_network_gradient_checks_passed','source_sha256':sha(__file__),'proposal_result_sha256':sha(source),'checks':checks,
        'scope':'Checks gradient/arithmeticimplementation, notvalidityoflinearproxyasapredictorofactualquality. CPUFP32only.'},indent=2)+'\n');print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
