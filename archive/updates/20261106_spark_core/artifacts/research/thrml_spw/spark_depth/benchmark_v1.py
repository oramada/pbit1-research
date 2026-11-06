"""Complete, checkpointed GPT2 benchmark for a calibration-selected K1 repair.

Compares the selected layout with original flat grouping at the exact same
per-tensor T budget. Model point reused within each case: explicit R1.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from . import causal_v1 as sampler
from .profile_calibration_v1 import policy
from .weight_survey_v1 import mapped

SEEDS=list(range(2026110801,2026110806))

def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
    import torch
    from transformers import AutoModelForCausalLM
    from research.thrml_spw.broad.r1_model_v1 import observations
    selection_file=OUT/'profile_extension_combined_v1.json';selection=json.loads(selection_file.read_text());selected=selection['selected_profile'];assert selected is not None,'No calibration-qualified candidate; do not score benchmark'
    audit_file=OUT/'profile_extension_combined_audit_v1.json';audit=json.loads(audit_file.read_text());assert audit['selected_profile']==selected and audit['result_sha256']==sha(selection_file)
    calibration=json.loads((OUT/'profile_calibration_v1/protocol.json').read_text());prefs=calibration['preferences'];data=json.loads((OUT/'benchmark_preparation_v1.json').read_text())
    cases=[{'name':'BF16','kind':'original'},{'name':'selected_mean','kind':'mean'}]
    cases += [{'name':f'{kind}_seed{seed}','kind':kind,'seed':seed}for seed in SEEDS for kind in ['selected','matched_flat']]
    protocol={'source_sha256':sha(__file__),'sampler_sha256':sha(sampler.__file__),'policy_source_sha256':sha(Path(__file__).with_name('profile_calibration_v1.py')),
        'observation_source_sha256':sha(Path('research/thrml_spw/broad/r1_model_v1.py')),'selection_sha256':sha(selection_file),'selection_audit_sha256':sha(audit_file),
        'benchmark_preparation_sha256':sha(OUT/'benchmark_preparation_v1.json'),'selected_profile':selected,'preferences':prefs,'data':data,'cases':cases,
        'checkpoint_sha256':sha(CHECKPOINT/'model.safetensors'),'runtime':{'torch':torch.__version__,'numpy':np.__version__,'GPU':torch.cuda.get_device_name()},'group':128,'GPU_allocator_cap':2*1024**3,
        'primary_gate':'Five selected-layout seeds: upper endpoint95%t4 CI(deltaNLL)<=ln1.02 relative to localBF16 on all283225targets. Matched-flat testedidentically and reportedregardless outcome. Allcasesandall586windowsrequired.',
        'scope':'PreviouslyinspectedcompleteWikiText2rawtest benchmarkre-evaluation; selectedonlyusingtrainingcalibration. R1idealbinomialcounts, notfullmodelnativeTHRMLorphysicalhardware. No newunexposedtestclaim.'}
    root=OUT/'benchmark_v1';root.mkdir(exist_ok=True);pfile=root/'protocol.json'
    if pfile.exists():assert json.loads(pfile.read_text())==protocol,'Frozen benchmark contract changed'
    else:atomic_json(pfile,protocol)
    if(root/'results.json').exists():return
    assert not(root/'failure.json').exists(),'Preserved benchmark failure requires a new version'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    params=dict(model.named_parameters());original={k:p.detach().cpu().clone()for k,p in params.items()};assert sum(p.numel()for p in params.values())==124439808
    assert model.lm_head.weight is model.transformer.wte.weight
    assert sha(data['tokens'])==data['tokens_sha256']and sha(data['layout'])==data['layout_sha256']
    with np.load(data['tokens'],allow_pickle=False)as d:tokens=d['tokens'].copy();offsets=d['offsets'].copy()
    layout=json.loads(Path(data['layout']).read_text());assert len(layout['windows'])==586
    started=time.monotonic();case_results=[]
    try:
        for case in cases:
            folder=root/case['name'];folder.mkdir(exist_ok=True);point_hash=hashlib.sha256();parameter_records=[]
            with torch.no_grad():
                for index,(name,param)in enumerate(params.items()):
                    orientation,T=policy(name,selected,prefs)
                    if case['kind']=='matched_flat':orientation='storage_flat'
                    if case['kind']=='original':value=original[name]
                    elif case['kind']=='mean':
                        w=original[name].float().numpy();o=w.T.copy()if orientation=='transposed_matrix'and w.ndim==2 else w
                        *_,mu,var=mapped(o,128);mu=mu.reshape(o.shape);mu=mu.T.copy()if orientation=='transposed_matrix'and w.ndim==2 else mu
                        value=torch.from_numpy(mu.astype(np.float32)).to(torch.bfloat16)
                    else:
                        sampler.SEED=case['seed'];value,record=sampler.sample_parameter(original[name],index,orientation,128,T,torch);record['name']=name;parameter_records.append(record)
                    point_hash.update(name.encode());point_hash.update(value.view(torch.uint16).numpy().tobytes());param.copy_(value.cuda())
            point=point_hash.hexdigest();parameters={'point_sha256':point,'records':parameter_records}
            paramfile=folder/'parameters.json'
            if paramfile.exists():assert json.loads(paramfile.read_text())==parameters
            else:atomic_json(paramfile,parameters)
            total_nll=0.;count=0;article_sums=np.zeros(62);article_counts=np.zeros(62,dtype=np.int64);peak=0
            for index,window in enumerate(layout['windows']):
                begin=int(offsets[window['article']]);ids=tokens[begin+window['context_start']:begin+window['context_stop']][None]
                file=folder/f'window{index:04d}.npz';receipt=folder/f'window{index:04d}.json';input_hash=sampler.digest(ids)
                if receipt.exists():
                    r=json.loads(receipt.read_text());assert r['input_sha256']==input_hash and r['point_sha256']==point and sha(file)==r['arrays_sha256']
                    with np.load(file,allow_pickle=False)as f:loss=f['token_NLL'].copy()
                else:
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():output=model(input_ids=torch.from_numpy(ids).cuda(),use_cache=False)
                    logits=output.logits.float().cpu().numpy();obs=observations({'task':'gpt2',**window},ids,None,logits);loss=obs['token_NLL']
                    tmp=file.with_suffix('.tmp')
                    with tmp.open('wb')as f:np.savez_compressed(f,**obs)
                    tmp.replace(file);r={'window':window,'input_sha256':input_hash,'point_sha256':point,'arrays_sha256':sha(file),'logits_sha256':sampler.digest(logits),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()};atomic_json(receipt,r);del output,logits,obs
                assert loss.shape==(window['target_stop']-window['target_start'],)and np.isfinite(loss).all()
                total_nll+=float(loss.sum());count+=len(loss);article_sums[window['article']]+=float(loss.sum());article_counts[window['article']]+=len(loss);peak=max(peak,r['GPU_peak_allocated_bytes'])
                atomic_json(root/'progress.json',{'case':case['name'],'window':index,'windows':586,'scored_targets':count,'total_targets':283225,'elapsed_seconds':time.monotonic()-started})
            assert count==283225
            summary={'case':case,'NLL':total_nll/count,'PPL':math.exp(total_nll/count),'targets':count,'article_NLL_sums':article_sums.tolist(),'article_target_counts':article_counts.tolist(),'GPU_peak_allocated_bytes':peak,'point_sha256':point}
            atomic_json(folder/'results.json',summary);case_results.append(summary);print(json.dumps({'completed':case['name'],'NLL':summary['NLL'],'PPL':summary['PPL']}),flush=True)
        baseline=case_results[0]['NLL'];stats={}
        for kind in ['selected','matched_flat']:
            values=np.array([r['NLL']for r in case_results if r['case']['kind']==kind]);assert len(values)==5;delta=values-baseline;half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
            stats[kind]={'PPL_ratio':float(np.exp(delta.mean())),'MC95_t_PPL_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'pass':bool(delta.mean()+half<=np.log(1.02))}
        atomic_json(root/'results.json',{'status':'complete_all12cases_fullbenchmark_pending_independent_audit','protocol_sha256':sha(pfile),'cases':case_results,'statistics':stats,'scope':protocol['scope']})
    except Exception as error:
        atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise

if __name__=='__main__':main()
