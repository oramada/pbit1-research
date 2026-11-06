"""Paired local causal diagnostics, exact ideal Bernoulli count law.

R1 reconstruction in conventional BF16 operators. Not THRML trajectories,
physical devices, R2 direct execution, or heldout effectiveness evidence.
"""
from pathlib import Path
import hashlib,json,time,math
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped
from .trace_v1 import score

SEED=2026110621
CHUNK=262144

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def sample_parameter(raw,index,layout,group,T,torch):
    w=raw.float().numpy();oriented=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w
    flat=oriented.reshape(-1);reconstruction=np.empty(flat.shape,np.float32)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,index]));counts_hash=hashlib.sha256();sse=expected=var_sse=0.
    for start in range(0,len(flat),CHUNK):
        stop=min(len(flat),start+CHUNK);codes,lo,span,mean,var=mapped(flat[start:stop],group)
        fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)
        prob=1/(1+np.exp(-2*fields.astype(np.float64)))
        c=rng.binomial(T,prob).astype(np.uint16);counts_hash.update(c.tobytes())
        lower=np.repeat(lo,group)[:len(c)].astype(np.float64);scale=np.repeat(span,group)[:len(c)].astype(np.float64)
        values=lower+scale*c.astype(np.float64)/T;reconstruction[start:stop]=values
        diff=values-mean;v=var/T;pq=prob*(1-prob)
        sse+=float(diff@diff);expected+=float(v.sum());var_sse+=float((2*v*v+scale**4*pq*(1-6*pq)/T**3).sum())
    value=reconstruction.reshape(oriented.shape)
    if layout=='transposed_matrix'and w.ndim==2:value=value.T.copy()
    value=torch.from_numpy(value.copy()).to(torch.bfloat16)
    return value,{'counts_uint16_sha256':counts_hash.hexdigest(),'BF16_sha256':digest(value.view(torch.uint16).numpy()),
        'SSE_about_finite_control_mean_before_FP32_BF16_rounding':sse,'expected_SSE':expected,'variance_of_SSE':var_sse,'samples':T,
        'layout':layout if w.ndim==2 else'storage_flat','group':group,'parameters':len(flat),'range_groups':math.ceil(len(flat)/group)}

def build_point(original,layout,group,torch):
    values={};records=[]
    for i,(name,w)in enumerate(original.items()):
        value,record=sample_parameter(w,i,layout,group,8192,torch);values[name]=value;record['name']=name;records.append(record)
    obs=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']for r in records)
    expected=sum(r['expected_SSE']for r in records);variance=sum(r['variance_of_SSE']for r in records)
    z=(obs-expected)/math.sqrt(variance);assert abs(z)<8,(layout,group,z)
    return values,{'parameters':records,'aggregate_SSE':obs,'aggregate_expected_SSE':expected,'aggregate_SSE_z':z,
        'total_Bernoulli_observations':8192*sum(r['parameters']for r in records),
        'logical_pbits':sum(r['parameters']for r in records),'controller_bits':12*sum(r['parameters']for r in records),
        'range_metadata_bits':64*sum(r['range_groups']for r in records)}

def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'causal_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    conditions=[{'name':'original_BF16','restore':'all'}, {'name':'ideal_counts_flat_G128','restore':'none'},
        {'name':'restore_token_embedding','restore':'transformer.wte.'},{'name':'restore_position_embedding','restore':'transformer.wpe.'},
        {'name':'restore_norms_and_biases','restore':'norms_biases'}]
    conditions += [{'name':f'restore_block_{i:02d}','restore':f'transformer.h.{i}.'}for i in range(12)]
    conditions += [{'name':'ideal_counts_transposed_G128','restore':'none','layout':'transposed_matrix','group':128},
        {'name':'ideal_counts_transposed_G32','restore':'none','layout':'transposed_matrix','group':32}]
    protocol={'source_sha256':sha(__file__),'mapping_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),
        'metric_source_sha256':sha(Path(__file__).with_name('trace_v1.py')),'prepared_sha256':sha(OUT/'prepared.json'),
        'conditions':conditions,'seed':SEED,'binomial_chunk':CHUNK,'samples':8192,'rows':list(range(8)),'input_length':129,
        'GPU_cap_bytes':2*1024**3,
        'scope':'Training-only causal diagnostic; one ideal independent binomial draw per parameter, reused across all eight inputs. Exact distribution of T independent Bernoulli observations, R1 BF16 reconstruction. No nativeTHRML, crossdeviceclaim, heldoutscore, confidenceintervalorqualitygate. Restored deterministic subsets are diagnostic departures from all-parameterK1; two layout candidates retainK1 andcount fullmetadata.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    params=dict(model.named_parameters());original={k:p.detach().cpu().clone()for k,p in params.items()}
    assert sum(p.numel()for p in params.values())==124439808
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as d:windows=d['input_ids'][:8,:129].copy()
    reference={};records=[];draw_reports={};draw=None;current_layout=None;current_group=None
    for condition in conditions:
        name=condition['name'];layout=condition.get('layout','storage_flat');group=condition.get('group',128)
        if name!='original_BF16'and(layout!=current_layout or group!=current_group):
            draw,draw_report=build_point(original,layout,group,torch);draw_reports[f'{layout}_G{group}']=draw_report;current_layout=layout;current_group=group
            (root/'sampling_audit.json').write_text(json.dumps(draw_reports,indent=2)+'\n')
        restored=0;point_hash=hashlib.sha256()
        with torch.no_grad():
            for param_name,p in params.items():
                rule=condition['restore'];restore=rule=='all'or(rule not in('none','norms_biases')and param_name.startswith(rule))or(rule=='norms_biases'and('.ln_'in param_name or param_name.endswith('.bias')))
                value=original[param_name]if restore else draw[param_name]
                if restore:restored+=p.numel()
                point_hash.update(param_name.encode());point_hash.update(value.view(torch.uint16).numpy().tobytes());p.copy_(value.cuda())
        condition_records=[]
        for row,ids in enumerate(windows):
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False)
            logits=out.logits.detach().float().cpu().numpy()
            if name=='original_BF16':reference[row]=logits.copy()
            metrics,nll=score(logits,reference[row],ids[None]);assert all(math.isfinite(v)for v in metrics.values())
            record={'condition':name,'row':row,'input_sha256':digest(ids),'logits_sha256':digest(logits),'metrics':metrics,
                'BF16_restored_parameters':restored,'point_sha256':point_hash.hexdigest(),'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            condition_records.append(record);records.append(record)
            with (root/f'{name}_row{row}_NLL.npy').open('xb')as f:np.save(f,nll)
            (root/f'{name}_row{row}.json').write_text(json.dumps(record,indent=2)+'\n')
            (root/'progress.json').write_text(json.dumps({'condition':name,'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n')
            print(json.dumps({'condition':name,'row':row,'mean_NLL':metrics['mean_NLL']}),flush=True)
            del out,logits
        mean=float(np.mean([r['metrics']['mean_NLL']for r in condition_records]));print(json.dumps({'finished_condition':name,'NLL':mean,'perplexity':math.exp(mean)}),flush=True)
    report={'status':'completed_training_only_single_seed_causal_and_layout_diagnostic','protocol_sha256':sha(root/'protocol.json'),
        'records':records,'draw_reports':draw_reports,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']}
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'forwards':len(records)}),flush=True)

if __name__=='__main__':main()
