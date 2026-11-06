"""Training-only activation-informed grouping for every original LFM3 parameter."""
from pathlib import Path
from collections import Counter
import json,sys,time,math
import numpy as np
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT,prepare,busy,point_hash
from .weight_survey_v1 import mapped

def score_weight(weight,layout,group,state,kind):
    shape=weight.shape;oriented=weight.T.copy()if layout=='transposed_matrix'else weight
    flat=oriented.reshape(-1);chunk=262144//group*group;total=0.;count=state['count']
    if kind=='embedding':mom=state['lookup_counts']/count
    elif kind=='conv':mom=state['input_sumsq'].reshape(-1)/count
    else:mom=state['input_sumsq']/count
    for start in range(0,len(flat),chunk):
        stop=min(start+chunk,len(flat));*_,variance=mapped(flat[start:stop],group)[3:]
        indices=np.arange(start,stop)
        if kind=='embedding':index=indices//shape[1]if layout=='storage_flat'else indices%shape[0]
        elif kind=='linear':index=indices%shape[1]if layout=='storage_flat'else indices//shape[0]
        else:index=indices
        total+=float(variance@mom[index])
    return total

def main():
    prepared,meta=prepare()
    if busy():raise RuntimeError('Keep GPU lane serial')
    import torch
    from transformers import AutoModelForCausalLM
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'lfm3_calibration_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    protocol={'source_sha256':sha(__file__),'model_preparation_sha256':sha(prepared),'data_preparation_sha256':sha(OUT/'lfm3_data_v1/results.json'),
        'rows':list(range(8)),'input_length':129,'samples_for_predicted_variance':32768,
        'selection':'Minimum conditional output variance per parameter. Tied embedding/head uses head input moments. Norm groups16or128; depthwiseconv groups7,16,128; matrices flatortransposedG128. Exactties choosefewer rangegroups.',
        'scope':'OriginalBF16checkpoint,training-onlyactivationmoments,allparameterpolicies. No stochasticmodeldraws,qualitylabels,heldoutforwards ortrainingupdates.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(ROOT,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
    modules={n:m for n,m in model.named_modules()if m._parameters.get('weight')is not None};assert len(modules)==287
    stats={};handles=[]
    def capture(name,module,args,output):
        x=args[0].detach();kind='linear'if isinstance(module,torch.nn.Linear)else'embedding'if isinstance(module,torch.nn.Embedding)else'conv'if isinstance(module,torch.nn.Conv1d)else'norm'
        if kind=='embedding':
            ids=x.cpu().numpy().reshape(-1);entry=stats.setdefault(name,{'kind':kind,'count':0,'lookup_counts':np.zeros(module.num_embeddings,np.int64)})
            entry['lookup_counts']+=np.bincount(ids,minlength=module.num_embeddings);entry['count']+=len(ids)
        elif kind=='conv':
            a=x.cpu().float().numpy().astype(np.float64);B,C,L=a.shape;kernel=module.kernel_size[0];sums=np.zeros((C,kernel))
            for k in range(kernel):
                lag=kernel-1-k
                if lag<L:sums[:,k]=(a[:,:,:L-lag]**2).sum(axis=(0,2))
            entry=stats.setdefault(name,{'kind':kind,'count':0,'input_sumsq':np.zeros_like(sums)});entry['count']+=B*L;entry['input_sumsq']+=sums
        else:
            if kind=='norm':x=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+module.variance_epsilon)).to(x.dtype)
            a=x.cpu().float().numpy().astype(np.float64).reshape(-1,x.shape[-1]);sums=(a*a).sum(0)
            entry=stats.setdefault(name,{'kind':kind,'count':0,'input_sumsq':np.zeros_like(sums)});entry['count']+=len(a);entry['input_sumsq']+=sums
    for name,module in modules.items():handles.append(module.register_forward_hook(lambda m,a,o,name=name:capture(name,m,a,o)))
    with np.load(OUT/'lfm3_data_v1/training_windows.npz')as f:ids=f['input_ids'][:8,:129].copy()
    peak=0
    try:
        with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
            for row,window in enumerate(ids):
                offloader.calls.clear();torch.cuda.reset_peak_memory_stats();out=model(input_ids=torch.from_numpy(window[None]).cuda(),use_cache=False)
                assert Counter(offloader.calls)==Counter(modules)and torch.isfinite(out.logits).all();peak=max(peak,torch.cuda.max_memory_allocated());del out
                (root/'progress.json').write_text(json.dumps({'stage':'training_activations','row':row,'elapsed_seconds':time.monotonic()-begin})+'\n');print('activation',row,flush=True)
    finally:
        for h in handles:h.remove()
    original_hash=point_hash(model);expected=json.loads((OUT/'lfm3_full_preflight_v2_audit.json').read_text())['weight_point_sha256'];assert original_hash==expected
    arrays={}
    for name,s in stats.items():
        for key,value in s.items():
            if isinstance(value,np.ndarray):arrays[name+'.'+key]=value
    with(root/'activation_moments.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    roles={}
    for name,m in modules.items():roles.setdefault(id(m.weight),[]).append(name)
    policies=[]
    for index,(name,param)in enumerate(model.named_parameters()):
        names=roles[id(param)];role='lm_head'if'lm_head'in names else names[0];state=stats[role];kind=state['kind'];w=param.detach().float().numpy()
        variants=[('storage_flat',128),('transposed_matrix',128)]if w.ndim==2 else [('storage_flat',g)for g in ([7,16,128]if kind=='conv'else[16,128])]
        values=[]
        for layout,group in variants:
            score=score_weight(w,layout,group,state,kind);values.append({'layout':layout,'group':group,'variance_coefficient':score,'range_groups':math.ceil(w.size/group)})
        selected=min(values,key=lambda x:(x['variance_coefficient'],x['range_groups']))
        policies.append({'name':name,'shape':list(w.shape),'parameters':param.numel(),'roles':names,'selected_role':role,'kind':kind,'variants':values,
            'selected_layout':selected['layout'],'selected_group':selected['group'],'samples':32768})
        (root/'progress.json').write_text(json.dumps({'stage':'range_selection','parameter':index,'name':name,'elapsed_seconds':time.monotonic()-begin})+'\n');print('range',index,name,flush=True)
    assert len(policies)==286 and sum(p['parameters']for p in policies)==1728801536
    (root/'results.json').write_text(json.dumps({'status':'LFM3_training_moments_and_all_parameter_range_policies_complete','protocol_sha256':sha(root/'protocol.json'),
        'activation_moments_sha256':sha(root/'activation_moments.npz'),'weight_point_sha256':original_hash,'policies':policies,'GPU_peak_allocated_bytes':peak,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']},indent=2)+'\n')

if __name__=='__main__':main()
