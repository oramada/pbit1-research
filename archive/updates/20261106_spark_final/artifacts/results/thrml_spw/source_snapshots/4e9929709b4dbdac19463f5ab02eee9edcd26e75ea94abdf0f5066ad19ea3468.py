"""Validate latency-capped vector-range proposals on CPU calibration data."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .counter_sampler_v2 import sample_parameter

def token_losses(logits,ids):
    result=[];z=logits[0,:-1];target=ids[1:]
    for start in range(0,len(z),16):
        block=z[start:start+16].astype(np.float64);maximum=block.max(1)
        normalizer=maximum+np.log(np.exp(block-maximum[:,None]).sum(1));result.extend(normalizer-block[np.arange(len(block)),target[start:start+16]])
    return np.asarray(result)

def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
    root=OUT/'cpu_vector_validation_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    proposal_file=OUT/'vector_range_allocation_v1/results.json';proposal=json.loads(proposal_file.read_text());preferences={r['weight']:r['preferred_by_local_variance']for r in json.loads((OUT/'activation_variance_v1/results.json').read_text())['operators']}
    policies={'uniform32768':{r['name']:{'layout':preferences.get(r['name'],'storage_flat'),'samples':32768}for r in json.loads((OUT/'loss_gradient_variance_v1/results.json').read_text())['parameters']}}
    for p in proposal['proposals']:policies[p['name']]={r['name']:r for r in p['parameters']}
    seeds=list(range(2026111311,2026111316));protocol={'source_sha256':sha(__file__),'sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v2.py')),'proposal_sha256':sha(proposal_file),'seeds':seeds,'policies':policies,
        'rows':list(range(16)),'length':257,'scope':'Diagnosticrows0:8 andcalibrationrows8:16 only. FP32CPUarithmetic withsampledBF16weights. No GPU,testdata,parameterupdates ordeploymentselection. One-pbitperparameter retained; vectorsusegroup16andallT<=32768. Pairedfive-seed comparisonsare descriptive; requirefutureprospectiveGPUconfirmation.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval()
    params=dict(model.named_parameters());original={n:p.detach().to(torch.bfloat16).clone()for n,p in params.items()}
    with torch.no_grad():
        for n,p in params.items():p.copy_(original[n].float())
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as f:windows=f['input_ids'][:16,:257].copy()
    records=[];costs={};cache={};last=None
    for profile,seed in [('baseline',None)]+[(name,s)for s in seeds for name in policies]:
        if seed!=last:cache={};last=seed
        point=hashlib.sha256();rows=[]
        with torch.no_grad():
            for index,(name,p)in enumerate(params.items()):
                if seed is None:value=original[name]
                else:
                    policy=policies[profile][name];layout=policy['layout'];T=policy['samples'];group=policy.get('group',128);key=(index,layout,T,group)
                    if key not in cache:cache[key]=sample_parameter(original[name],index,layout,group,T,torch,seed)
                    value,r=cache[key];rows.append({'name':name,**r})
                point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes());p.copy_(value.float())
        condition=profile if seed is None else f'{profile}_seed{seed}'
        if rows:
            (root/f'{condition}_parameters.json').write_text(json.dumps(rows,indent=2)+'\n');costs[profile]=sum(r['parameters']*r['samples']for r in rows)
        for row,ids in enumerate(windows):
            with torch.inference_mode():output=model(input_ids=torch.from_numpy(ids[None]),use_cache=False)
            logits=output.logits.numpy();loss=token_losses(logits,ids);assert np.isfinite(loss).all()
            with(root/f'{condition}_row{row}.npy').open('xb')as f:np.save(f,loss)
            r={'profile':profile,'seed':seed,'row':row,'mean_NLL':float(loss.mean()),'targets':len(loss),'input_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'point_sha256':point.hexdigest()};records.append(r)
            (root/f'{condition}_row{row}.json').write_text(json.dumps(r,indent=2)+'\n');(root/'progress.json').write_text(json.dumps({'condition':condition,'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n');del output,logits
        print(json.dumps({'completed':condition,'elapsed_seconds':time.monotonic()-begin}),flush=True)
    summaries={}
    for split,selected_rows in [('diagnostic',range(8)),('calibration',range(8,16))]:
        base=np.mean([r['mean_NLL']for r in records if r['profile']=='baseline'and r['row']in selected_rows]);summaries[split]={}
        uniform=np.array([np.mean([r['mean_NLL']for r in records if r['profile']=='uniform32768'and r['seed']==seed and r['row']in selected_rows])for seed in seeds])
        for name in policies:
            values=np.array([np.mean([r['mean_NLL']for r in records if r['profile']==name and r['seed']==seed and r['row']in selected_rows])for seed in seeds]);delta=values-base;paired=values-uniform;half=2.7764451051977987*paired.std(ddof=1)/np.sqrt(5)
            summaries[split][name]={'PPL_ratio_to_CPU_baseline':float(np.exp(delta.mean())),'paired_PPL_ratio_to_uniform':float(np.exp(paired.mean())),
                'paired_MC95_t_PPL_ratio_to_uniform':np.exp([paired.mean()-half,paired.mean()+half]).tolist(),'cost_ratio_to_uniform':costs[name]/costs['uniform32768']}
    assert not torch.cuda.is_initialized()
    (root/'results.json').write_text(json.dumps({'status':'completed_CPU_proposal_validation_no_gpu_confirmation','protocol_sha256':sha(root/'protocol.json'),'records':records,'costs':costs,'summaries':summaries,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin},indent=2)+'\n');print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
