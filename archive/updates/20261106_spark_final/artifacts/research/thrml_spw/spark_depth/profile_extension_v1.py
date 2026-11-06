"""Five additional calibration seeds for the two strongest first-stage policies.

Keep the first five-seed failures unchanged; only the combined ten-seed audit
can select a policy for the new benchmark.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from . import causal_v1 as sampler
from .weight_survey_v1 import mapped
from .trace_v1 import score

PROFILES=['operator_body_T32768','operator_all_T32768']
SEEDS=list(range(2026110671,2026110676))

def policy(name,profile,preferences):
    layout='transposed_matrix'if name=='transformer.wte.weight'else'storage_flat'
    if profile!='embedding_only_T8192':layout=preferences.get(name,'storage_flat')
    T=32768 if profile=='operator_all_T32768'or(profile=='operator_body_T32768'and name!='transformer.wte.weight')else 8192
    return layout,T

def ledger(rows):
    n=sum(r['parameters']for r in rows);obs=sum(r['parameters']*r['samples']for r in rows)
    v=sum(r['variance_of_SSE']for r in rows);sse=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']for r in rows);expected=sum(r['expected_SSE']for r in rows)
    z=(sse-expected)/math.sqrt(v);assert abs(z)<8
    return {'logical_pbits':n,'observations':obs,'mean_samples':obs/n,'versus_uniform_T8192_observations':obs/(n*8192),
        'control_bits':12*n,'range_bits':64*sum(r['range_groups']for r in rows),'policy_bits':17*len(rows),
        'ideal_control_range_state_policy_bits_per_weight':13+(64*sum(r['range_groups']for r in rows)+17*len(rows))/n,'SSE_z':z}

def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'profile_extension_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    pref_file=OUT/'activation_variance_v1/results.json';pref_report=json.loads(pref_file.read_text());preferences={r['weight']:r['preferred_by_local_variance']for r in pref_report['operators']}
    protocol={'parent_five_seed_result_sha256':sha(OUT/'profile_calibration_v1/results.json'),'source_sha256':sha(__file__),'sampler_sha256':sha(sampler.__file__),'preferences_sha256':sha(pref_file),'prepared_sha256':sha(OUT/'prepared.json'),
        'profiles':PROFILES,'seeds':SEEDS,'group':128,'diagnostic_rows':list(range(8)),'calibration_rows':list(range(8,16)),'input_length':257,
        'selection':'Cheapest observations among profiles whose calibration deltaNLL upper95% t4 endpoint <= ln1.02; tie break lower mean deltaNLL. No passingprofile => no selection. Selection oncalibration isnotconfirmation.',
        'preferences':preferences,'policy_storage_assumption':'one orientationbit and16bitTcode per canonicaltensor; shape/architecture shared with baseline.',
        'scope':'R1 idealindependentbinomial K1allparameters. Sampleparameterkey(seed,index,layout,T); identical parameters reused across profiles, paired comparison. No confirmationrows16:32 or officialtest forwards. Allseedsrequired.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    params=dict(model.named_parameters());original={k:p.detach().cpu().clone()for k,p in params.items()};assert sum(p.numel()for p in params.values())==124439808
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as d:windows=d['input_ids'][:16,:257].copy()
    reference={};records=[];ledgers={};parameter_artifacts={};cache={};last_seed=None
    conditions=[('BF16',None),('finite_mean_operator',None)]+[(profile,seed)for seed in SEEDS for profile in PROFILES]
    for profile,seed in conditions:
        name=profile if seed is None else f'{profile}_seed{seed}'
        if seed!=last_seed:cache={};last_seed=seed
        rows=[];point_hash=hashlib.sha256()
        with torch.no_grad():
            for index,(key,p)in enumerate(params.items()):
                if profile=='BF16':value=original[key]
                elif profile=='finite_mean_operator':
                    layout,_=policy(key,'operator_T8192',preferences);w=original[key].float().numpy();o=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w
                    *_,mean,var=mapped(o,128);mean=mean.reshape(o.shape);mean=mean.T.copy()if layout=='transposed_matrix'and w.ndim==2 else mean
                    value=torch.from_numpy(mean.astype(np.float32)).to(torch.bfloat16)
                else:
                    layout,T=policy(key,profile,preferences);assert 1<=T<=65535;cachekey=(index,layout,T)
                    if cachekey not in cache:
                        sampler.SEED=seed;value,record=sampler.sample_parameter(original[key],index,layout,128,T,torch);record['name']=key;cache[cachekey]=(value,record)
                    value,record=cache[cachekey];rows.append(record)
                point_hash.update(key.encode());point_hash.update(value.view(torch.uint16).numpy().tobytes());p.copy_(value.cuda())
        if rows:
            ledgers[name]=ledger(rows);parameter_artifacts[name]=rows
            (root/f'{name}_parameters.json').write_text(json.dumps(rows,indent=2)+'\n')
        for row,ids in enumerate(windows):
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False)
            logits=out.logits.detach().float().cpu().numpy()
            if profile=='BF16':reference[row]=logits.copy()
            metrics,nll=score(logits,reference[row],ids[None]);assert all(math.isfinite(v)for v in metrics.values())
            r={'condition':name,'profile':profile,'seed':seed,'row':row,'split':'diagnostic'if row<8 else'calibration','metrics':metrics,
                'input_sha256':sampler.digest(ids),'logits_sha256':sampler.digest(logits),'point_sha256':point_hash.hexdigest(),'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            records.append(r);(root/f'{name}_row{row}.json').write_text(json.dumps(r,indent=2)+'\n')
            with(root/f'{name}_row{row}_NLL.npy').open('xb')as f:np.save(f,nll)
            (root/'progress.json').write_text(json.dumps({'condition':name,'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n');del out,logits
        print(json.dumps({'condition_complete':name,'elapsed_seconds':time.monotonic()-begin}),flush=True)
    summaries={};passing=[]
    for split in ['diagnostic','calibration']:
        subset=[r for r in records if r['split']==split];base=np.mean([r['metrics']['mean_NLL']for r in subset if r['profile']=='BF16']);summaries[split]={'BF16_NLL':float(base),'profiles':{}}
        for profile in PROFILES:
            means=np.array([np.mean([r['metrics']['mean_NLL']for r in subset if r['profile']==profile and r['seed']==seed])for seed in SEEDS]);delta=means-base;half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5);passed=bool(delta.mean()+half<=np.log(1.02))
            cost=ledgers[f'{profile}_seed{SEEDS[0]}']['versus_uniform_T8192_observations']
            summaries[split]['profiles'][profile]={'mean_PPL_ratio':float(np.exp(delta.mean())),'MC95_t_PPL_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'passes_calibration_rule':passed,'relative_observation_cost':cost,'seed_NLL':means.tolist()}
            if split=='calibration'and passed:passing.append((cost,float(delta.mean()),profile))
    result={'status':'completed_additional_five_seeds_selection_requires_combined_ten_seed_audit','protocol_sha256':sha(root/'protocol.json'),'records':records,'ledgers':ledgers,
        'summaries':summaries,'selected_profile':min(passing)[2]if passing else None,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summaries':summaries,'selected_profile':result['selected_profile']},indent=2),flush=True)

if __name__=='__main__':main()
