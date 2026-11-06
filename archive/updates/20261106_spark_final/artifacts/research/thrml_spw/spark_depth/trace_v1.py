"""Memory-capped GB10 diagnostic traces on eight training-only windows.

Gaussian weight noise is a labelled variance-matched proxy, never a native
p-bit result. Preserve exact baseline activations for paired neuron diagnostics.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped


def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def tensor(x,torch):
    if isinstance(x,(tuple,list)):x=x[0]
    if not torch.is_tensor(x):raise TypeError(type(x).__name__)
    return x.detach().float().cpu().numpy()

def channel_stats(a,b):
    x=a.reshape(-1,a.shape[-1]).astype(np.float64);y=b.reshape(-1,b.shape[-1]).astype(np.float64);d=x-y
    columns=np.stack([x.sum(0),(x*x).sum(0),y.sum(0),(y*y).sum(0),d.sum(0),(d*d).sum(0),np.abs(x).max(0),np.abs(d).max(0),(x*y).sum(0)])
    idx=np.unique(np.linspace(0,x.shape[1]-1,min(32,x.shape[1]),dtype=int));sd=d[:,idx]
    return columns,idx,sd.T@sd,{'rows':len(x),'channels':x.shape[1],'RMSE':float(np.sqrt(np.mean(d*d))),
        'reference_RMS':float(np.sqrt(np.mean(y*y))),'max_abs_error':float(np.abs(d).max())}

def score(logits,reference,ids):
    z=logits[0,:-1];ref=reference[0,:-1];target=ids[0,1:];nll=[];kl=[];flips=[];margin=[]
    for start in range(0,len(z),8):
        x=z[start:start+8].astype(np.float64);b=ref[start:start+8].astype(np.float64);labels=target[start:start+8]
        lx=x-np.max(x,axis=1,keepdims=True);lb=b-np.max(b,axis=1,keepdims=True)
        lx-=np.log(np.exp(lx).sum(1,keepdims=True));lb-=np.log(np.exp(lb).sum(1,keepdims=True))
        nll.extend(-lx[np.arange(len(x)),labels]);kl.extend((np.exp(lb)*(lb-lx)).sum(1));flips.extend(x.argmax(1)!=b.argmax(1))
        top=np.partition(b,-2,axis=1)[:,-2:];margin.extend(top.max(1)-top.min(1))
    return {'mean_NLL':float(np.mean(nll)),'top1_accuracy':float(np.mean(z.argmax(1)==target)),
        'reference_to_variant_KL':float(np.mean(kl)),'argmax_flip_fraction':float(np.mean(flips)),
        'mean_reference_top2_margin':float(np.mean(margin)),'targets':len(target)},np.asarray(nll,np.float64)

def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and 'GB10'in torch.cuda.get_device_name()
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/total))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'trace_v1';root.mkdir(exist_ok=False)
    cfg={'status':'training_only_exploratory_neuron_trace_frozen_before_forwards','source_sha256':sha(__file__),
        'prepared_sha256':sha(OUT/'prepared.json'),'mapping_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),
        'rows':list(range(8)),'input_length':129,'points':['original_BF16','finite_mean_G128','Gaussian_proxy_G128_T8192_seed2026110611'],
        'Gaussian_seed':2026110611,'sampling_budget_for_proxy':8192,'GPU_allocator_cap_bytes':2*1024**3,
        'scope':'LocalGB10, trainingarticlesonly, one labelledGaussianproxy. No nativepbitsamples,heldoutscore,confidenceinterval,trainingupdateorqualitygate.'}
    (root/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');begin=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    params=dict(model.named_parameters());assert sum(p.numel()for p in params.values())==124439808
    original={name:p.detach().cpu().clone()for name,p in params.items()};modules=dict(model.named_modules())
    names=['transformer.wte','transformer.wpe','transformer.drop','transformer.ln_f','lm_head']
    for layer in range(12):
        prefix=f'transformer.h.{layer}';names.extend([prefix,prefix+'.ln_1',prefix+'.attn.c_attn',prefix+'.attn.c_proj',prefix+'.ln_2',prefix+'.mlp.c_fc',prefix+'.mlp.act',prefix+'.mlp.c_proj'])
    captured={};handles=[]
    for name in names:
        def hook(module,args,output,name=name):captured[name]=tensor(output,torch)
        handles.append(modules[name].register_forward_hook(hook))
        if name in [f'transformer.h.{i}'for i in range(12)]:
            def before(module,args,name=name):captured[name+'.input']=tensor(args[0],torch)
            handles.append(modules[name].register_forward_pre_hook(before))
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as a:all_ids=a['input_ids'][:8,:129].copy()
    records=[];points=[]
    try:
        for point in cfg['points']:
            point_hash=hashlib.sha256()
            with torch.no_grad():
                for index,(name,p)in enumerate(params.items()):
                    raw=original[name]
                    if point=='original_BF16':value=raw
                    else:
                        w=raw.float().numpy();_,_,_,mean,var=mapped(w,128)
                        if point.startswith('Gaussian'):
                            rng=np.random.default_rng(np.random.SeedSequence([cfg['Gaussian_seed'],index]))
                            mean=mean+np.sqrt(var/8192)*rng.standard_normal(mean.shape)
                        value=torch.from_numpy(mean.astype(np.float32).reshape(w.shape)).to(dtype=torch.bfloat16)
                    point_hash.update(name.encode());point_hash.update(value.view(torch.uint16).numpy().tobytes());p.copy_(value.to(device='cuda'))
            points.append({'name':point,'weight_point_sha256':point_hash.hexdigest()})
            directory=root/point;directory.mkdir()
            for row,ids in enumerate(all_ids):
                captured.clear();torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False,output_attentions=True)
                logits=out.logits.detach().float().cpu().numpy()
                assert len(out.attentions)==12
                for layer,attention in enumerate(out.attentions):captured[f'ATTENTION_{layer}']=tensor(attention,torch)
                captured['LOGITS']=logits;assert all(np.isfinite(a).all()for a in captured.values())
                basefile=root/'original_BF16'/f'row{row}.npz';stats={};scalar={};attention_stats=[]
                if point=='original_BF16':
                    with basefile.open('xb')as f:np.savez_compressed(f,**captured)
                    baseline=captured
                else:
                    with np.load(basefile,allow_pickle=False)as f:baseline={k:f[k].copy()for k in f.files}
                for name in names+[f'transformer.h.{i}.input'for i in range(12)]:
                    a=captured[name];b=baseline[name];assert a.shape==b.shape
                    columns,idx,cov,summary=channel_stats(a,b)
                    stats[name+'_channel_sums']=columns;stats[name+'_covariance_indices']=idx;stats[name+'_error_second_moment']=cov
                    scalar[name]=summary
                for layer in range(12):
                    a=captured[f'ATTENTION_{layer}'].astype(np.float64);b=baseline[f'ATTENTION_{layer}'].astype(np.float64)
                    a=np.maximum(a,1e-30);b=np.maximum(b,1e-30)
                    ent=-(a*np.log(a)).sum(-1).mean(axis=(0,2));kl=(b*(np.log(b)-np.log(a))).sum(-1).mean(axis=(0,2))
                    err=np.sqrt(((a-b)**2).mean(axis=(0,2,3)));attention_stats.append(np.stack([ent,kl,err,a.max(-1).mean(axis=(0,2))]))
                stats['attention_head_statistics']=np.stack(attention_stats)
                metrics,nll=score(logits,baseline['LOGITS'],ids[None]);stats['token_NLL']=nll
                statfile=directory/f'row{row}_statistics.npz'
                with statfile.open('xb')as f:np.savez_compressed(f,**stats)
                record={'point':point,'row':row,'input_sha256':arrsha(ids),'statistics_file':str(statfile),
                    'statistics_sha256':sha(statfile),'baseline_file':str(basefile),'baseline_sha256':sha(basefile),
                    'logits_sha256':arrsha(logits),'metrics':metrics,'modules':scalar,
                    'torch_peak_allocated_bytes_this_forward':torch.cuda.max_memory_allocated()}
                (directory/f'row{row}.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
                (root/'progress.json').write_text(json.dumps({'point':point,'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n')
                print(json.dumps({'point':point,'row':row,'metrics':metrics}),flush=True)
                del out,baseline,stats;captured.clear()
    finally:
        for handle in handles:handle.remove()
    result={'status':'complete_training_only_three_point_GB10_neuron_head_layer_trace','protocol_sha256':sha(root/'protocol.json'),
        'points':points,'records':records,'elapsed_seconds':time.monotonic()-begin,
        'channel_sum_rows':['sum_variant','sumsq_variant','sum_reference','sumsq_reference','sum_error','sumsq_error','maxabs_variant','maxabs_error','sum_crossproduct'],
        'attention_stat_rows':['entropy','reference_to_variant_KL','probability_RMSE','mean_max_attention_probability'],
        'scope':cfg['scope']+' Fullmodulechannelstatistics and32selectedchannelerrorsecondmoments; not full dense channelcovariance. Rawbaseline activations retained; Gaussianpointisnotnativepbitexecution.'}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'forwards':len(records)}),flush=True)

if __name__=='__main__':main()
