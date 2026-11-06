"""Refine small 1D parameter ranges to avoid long-tail sampling latency."""
from pathlib import Path
import json,time,math
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped

def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
    root=OUT/'vector_range_allocation_v1';root.mkdir(exist_ok=False);begin=time.monotonic();prior_file=OUT/'loss_gradient_variance_v1/results.json';prior=json.loads(prior_file.read_text())
    (root/'protocol.json').write_text(json.dumps({'source_sha256':sha(__file__),'gradient_results_sha256':sha(prior_file),'rows':list(range(8)),'input_length':129,
        'groups_for_1D':[16,32,64,128],'2D_group':128,'sample_cap':32768,'scope':'CPUFP32training-onlygradientproxy, no weightupdates orqualityclaim. Shrinkrangeson1Dnorm/biasvectors whilekeepingonepbitperscalar andcountingmetadata.'},indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval()
    with torch.no_grad():
        for p in model.parameters():p.copy_(p.to(torch.bfloat16).float())
    params=dict(model.named_parameters());vectors={name:p for name,p in params.items()if p.ndim==1};cache={};scores={}
    for name,p in vectors.items():
        w=p.detach().numpy();cache[name]={g:mapped(w,g)[4].astype(np.float32)for g in [16,32,64,128]};scores[name]={g:0. for g in cache[name]}
    with np.load(OUT/'training_diagnostic_windows.npz')as f:ids=f['input_ids'][:8,:129].copy()
    for row,x in enumerate(ids):
        model.zero_grad(set_to_none=True);o=model(input_ids=torch.from_numpy(x[None]),labels=torch.from_numpy(x[None]),use_cache=False);o.loss.backward()
        for name,p in vectors.items():
            grad=p.grad.detach().numpy().astype(np.float64)
            for g,v in cache[name].items():scores[name][g]+=float(np.sum(grad*grad*v))/8
        del o
        (root/'progress.json').write_text(json.dumps({'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n')
    old={r['name']:r for r in prior['parameters']}
    for name,value in scores.items():np.testing.assert_allclose(value[128],old[name]['storage_S'],rtol=1e-5,atol=1e-20)
    names=list(params);N=np.array([p.numel()for p in params.values()]);target=prior['allocation_proposals'][0]['target_proxy_variance'];proposals=[]
    current={r['weight']:r['preferred_by_local_variance']for r in json.loads((OUT/'activation_variance_v1/results.json').read_text())['operators']}
    for kind in ['existing_directions','gradient_proxy_directions']:
        values=[];layouts=[]
        for name,p in params.items():
            if p.ndim==1:values.append(scores[name][16]);layouts.append('storage_flat')
            else:
                layout=current.get(name,'storage_flat')if kind=='existing_directions'else min(['storage_flat','transposed_matrix'],key=lambda l:old[name]['storage_S'if l=='storage_flat'else'transposed_S'])
                layouts.append(layout);values.append(old[name]['storage_S'if layout=='storage_flat'else'transposed_S'])
        S=np.array(values);T=np.full(len(names),1024,np.int64)
        while np.sum(S/T)>target:
            gains=np.where(T<32768,S/(2*N*T.astype(float)**2),-np.inf);i=int(np.argmax(gains))
            if not np.isfinite(gains[i]):break
            T[i]*=2
        rows=[{'name':name,'parameters':int(N[i]),'group':16 if params[name].ndim==1 else 128,'layout':layouts[i],'samples':int(T[i]),'proxy_coefficient':float(S[i])}for i,name in enumerate(names)]
        groups=sum(math.ceil(r['parameters']/r['group'])for r in rows)
        proposals.append({'name':kind+'_vectorG16','proxy_constraint_satisfied':bool(np.sum(S/T)<=target),'proxy_variance':float(np.sum(S/T)),'target_proxy_variance':target,
            'observation_cost_ratio':float(np.sum(N*T)/(N.sum()*32768)),'maximum_samples':int(T.max()),'range_metadata_bits':64*groups,
            'extra_range_bytes_over_G128':8*(groups-sum(math.ceil(r['parameters']/128)for r in rows)),'parameters':rows})
    result={'status':'completed_latency_capped_vector_range_proposals','protocol_sha256':sha(root/'protocol.json'),'vector_parameters':sum(p.numel()for p in vectors.values()),
        'vector_scores':scores,'proposals':proposals,'elapsed_seconds':time.monotonic()-begin,'scope':'Traininggradientproxyonly; futureactualqualityvalidation required.'}
    assert not torch.cuda.is_initialized();(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in p.items()if k!='parameters'}for p in proposals],indent=2))

if __name__=='__main__':main()
