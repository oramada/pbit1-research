"""Training-only first-order loss fluctuation and temporal-budget proposals.

Squared gradients of article-mean loss are a proxy, not a Hessian, expected
loss increase, per-token Fisher matrix, or validated allocation policy.
"""
from pathlib import Path
import json,hashlib,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped

def allocate(S,N,target):
    budgets=np.full(len(S),1024,np.int64)
    while float(np.sum(S/budgets))>target:
        gains=np.where(budgets<131072,S/(2*N*budgets.astype(float)**2),-np.inf)
        index=int(np.argmax(gains))
        if not np.isfinite(gains[index])or gains[index]<=0:return budgets,False
        budgets[index]*=2
    return budgets,True

def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
    available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    assert available>6*1024**3,'Insufficient CPU memory for cached diagnostic variances'
    root=OUT/'loss_gradient_variance_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    protocol={'source_sha256':sha(__file__),'mapping_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),'prepared_sha256':sha(OUT/'prepared.json'),
        'rows':list(range(8)),'length':129,'precision':'CPU FP32 arithmetic with BF16-rounded original weights','no_parameter_updates':True,
        'proxy':'Average over eight article-mean loss gradients: S_g=sum_i E[g_i^2]*VarCoefficient_i. Predicts first-order scalar-loss fluctuation, notmeanloss degradation.',
        'allocation':'Greedy powers-of-two upgrades from1024to131072 to match proxyvariance of existingoperatorlayout atuniform32768; choose highestvariance-reduction peraddedobservation.',
        'scope':'Diagnosticproposal only. No heldoutforwards; no fine-tuning. Calibration lacks longpositions/rarevocabulary; gradients and proxy alone do not establish transfer or quality.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval()
    with torch.no_grad():
        for p in model.parameters():p.copy_(p.to(torch.bfloat16).float())
    params=dict(model.named_parameters());assert sum(p.numel()for p in params.values())==124439808
    old=json.loads(Path('results/thrml_spw/broad_verification_20261105/remote_metadata/additional_parameter_packages_v1/gpt2/package.json').read_text())['canonical_parameters']
    cache={};names=list(params);N=np.array([params[n].numel()for n in names]);sum_S=np.zeros((len(names),2));sum_bias=np.zeros_like(sum_S);axes={};losses=[]
    for index,(name,p)in enumerate(params.items()):
        original=p.detach();assert hashlib.sha256(original.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()).hexdigest()==old[index]['original_BF16_raw_sha256']
        w=original.numpy();cache[name]=[]
        for layout in ['storage_flat','transposed_matrix']:
            a=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w
            _,_,_,mean,variance=mapped(a,128);mean=mean.reshape(a.shape);variance=variance.reshape(a.shape)
            if layout=='transposed_matrix'and w.ndim==2:mean=mean.T;variance=variance.T
            rounded=torch.from_numpy(mean.astype(np.float32).copy()).to(torch.bfloat16).float().numpy()
            cache[name].append((variance.astype(np.float32).copy(),(rounded-w).copy()))
        (root/'progress.json').write_text(json.dumps({'stage':'cache_controls','parameter':name,'elapsed_seconds':time.monotonic()-begin})+'\n')
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as d:ids=d['input_ids'][:8,:129].copy()
    for row,window in enumerate(ids):
        model.zero_grad(set_to_none=True)
        out=model(input_ids=torch.from_numpy(window[None]),labels=torch.from_numpy(window[None]),use_cache=False)
        assert torch.isfinite(out.loss);out.loss.backward();losses.append(float(out.loss.detach()))
        for index,(name,p)in enumerate(params.items()):
            if p.grad is None:continue
            g=p.grad.detach().numpy();g64=g.astype(np.float64);square=g64*g64
            for layout,(var,bias)in enumerate(cache[name]):
                terms=square*var.astype(np.float64);sum_S[index,layout]+=float(terms.sum());sum_bias[index,layout]+=float(np.sum(g64*bias))
                if g.ndim==2:
                    for axis in [0,1]:
                        key=f'p{index:03d}_layout{layout}_axis{axis}';value=terms.sum(axis=1-axis);axes[key]=axes.get(key,0)+value/8
        del out
        (root/'progress.json').write_text(json.dumps({'stage':'CPU_backward','row':row,'elapsed_seconds':time.monotonic()-begin})+'\n')
    sum_S/=8;sum_bias/=8
    pref=json.loads((OUT/'activation_variance_v1/results.json').read_text())['operators'];current_map={r['weight']:r['preferred_by_local_variance']for r in pref}
    current=np.array([int(current_map.get(n,'storage_flat')=='transposed_matrix'and params[n].ndim==2)for n in names])
    best=np.argmin(sum_S,axis=1);baseS=sum_S[np.arange(len(names)),current];target=float(baseS.sum()/32768);reports=[]
    for name,layout in [('existing_directions',current),('gradient_proxy_directions',best)]:
        S=sum_S[np.arange(len(names)),layout];T,feasible=allocate(S,N,target)
        reports.append({'name':name,'proxy_constraint_satisfied':feasible,'proxy_variance':float(np.sum(S/T)),'target_proxy_variance':target,
            'observations':int(np.sum(N*T)),'versus_uniform_32768_cost':float(np.sum(N*T)/(np.sum(N)*32768)),
            'parameters':[{'name':n,'layout':'transposed_matrix'if int(layout[i])else'storage_flat','samples':int(T[i]),'parameters':int(N[i]),'proxy_coefficient':float(S[i])}for i,n in enumerate(names)]})
    with(root/'channel_proxy_statistics.npz').open('xb')as f:np.savez_compressed(f,**axes)
    result={'status':'completed_training_only_CPU_loss_gradient_proxy','protocol_sha256':sha(root/'protocol.json'),'losses_FP32':losses,
        'parameters':[{'name':n,'elements':int(N[i]),'storage_S':float(sum_S[i,0]),'transposed_S':float(sum_S[i,1]),'storage_first_order_mean_control_shift':float(sum_bias[i,0]),'transposed_first_order_mean_control_shift':float(sum_bias[i,1])}for i,n in enumerate(names)],
        'allocation_proposals':reports,'channel_statistics_sha256':sha(root/'channel_proxy_statistics.npz'),'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']}
    assert not torch.cuda.is_initialized();(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'allocation_cost_ratios':{r['name']:r['versus_uniform_32768_cost']for r in reports}}))

if __name__=='__main__':main()
