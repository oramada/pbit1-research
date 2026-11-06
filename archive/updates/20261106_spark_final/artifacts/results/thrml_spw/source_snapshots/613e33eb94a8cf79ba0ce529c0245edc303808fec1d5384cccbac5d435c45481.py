"""Prospective five-seed comparison on disjoint, reserved training articles.

Selected on diagnostic rows0:8. This confirms a local repair, not universal
model quality and not the official WikiText2 validation/test benchmark.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from . import causal_v1 as sampler
from .trace_v1 import score


def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'confirm_layout_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    seeds=list(range(2026110631,2026110636));layouts=['storage_flat','transposed_matrix']
    protocol={'source_sha256':sha(__file__),'sampler_sha256':sha(sampler.__file__),'metric_sha256':sha(score.__code__.co_filename),
        'selection_results_sha256':sha(OUT/'causal_v1/results.json'),'prepared_sha256':sha(OUT/'prepared.json'),
        'seeds':seeds,'group':128,'samples':8192,'layouts':layouts,'input_length':257,
        'calibration_rows':list(range(8,16)),'confirmation_rows':list(range(16,32)),
        'frozen_confirmation_gate':'On fixed confirmation articles, upper endpoint of two-sided95% Student t interval for seed-mean deltaNLL <= log(1.02). df4 critical2.7764451051977987. Allfive seeds required. CI concerns MonteCarlo variation only, not unseenarticle ormodel generalization.',
        'scope':'R1 ideal independent binomial counts on DGXSpark. Same124439808logicalpbits andT8192. No THRML/hardware/nonideality/R2 claim. All reserved articles are fromtraining partition; no officialbenchmarktestclaim.',
        'seed_parameterization':'Set imported sampler.SEED to each frozen seed before sampling. Perparameter RNG SeedSequence([seed,index]), chunk262144. No source mutation.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    params=dict(model.named_parameters());original={k:p.detach().cpu().clone()for k,p in params.items()};assert sum(p.numel()for p in params.values())==124439808
    with np.load(OUT/'training_diagnostic_windows.npz',allow_pickle=False)as data:windows=data['input_ids'][8:32,:257].copy()
    baselines={};records=[];audits={}
    conditions=[('BF16',None,None)]+[(f'{layout}_seed{seed}',layout,seed)for seed in seeds for layout in layouts]
    for name,layout,seed in conditions:
        if seed is None:point=original
        else:
            sampler.SEED=seed;point,audit=sampler.build_point(original,layout,128,torch);audits[name]=audit
            (root/f'{name}_sampling.json').write_text(json.dumps(audit,indent=2)+'\n')
        point_hash=hashlib.sha256()
        with torch.no_grad():
            for key,p in params.items():
                value=point[key];point_hash.update(key.encode());point_hash.update(value.view(torch.uint16).numpy().tobytes());p.copy_(value.cuda())
        for localrow,ids in enumerate(windows):
            row=localrow+8;torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False)
            logits=out.logits.detach().float().cpu().numpy()
            if seed is None:baselines[row]=logits.copy()
            metrics,nll=score(logits,baselines[row],ids[None]);assert all(math.isfinite(x)for x in metrics.values())
            record={'condition':name,'layout':layout,'seed':seed,'row':row,'split':'calibration'if row<16 else'confirmation',
                'metrics':metrics,'input_sha256':sampler.digest(ids),'logits_sha256':sampler.digest(logits),
                'point_sha256':point_hash.hexdigest(),'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            records.append(record);(root/f'{name}_row{row}.json').write_text(json.dumps(record,indent=2)+'\n')
            with(root/f'{name}_row{row}_NLL.npy').open('xb')as f:np.save(f,nll)
            (root/'progress.json').write_text(json.dumps({'condition':name,'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n')
            del out,logits
        print(json.dumps({'finished':name,'elapsed_seconds':time.monotonic()-begin}),flush=True)
        if seed is not None:del point
    result={'status':'complete_all_five_seeds_both_layouts_reserved_training_articles','protocol_sha256':sha(root/'protocol.json'),
        'records':records,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope'],'summaries':{}}
    for split in ['calibration','confirmation']:
        selected=[r for r in records if r['split']==split];baseline=np.mean([r['metrics']['mean_NLL']for r in selected if r['seed']is None]);result['summaries'][split]={'BF16_NLL':float(baseline),'BF16_PPL':math.exp(baseline),'layouts':{}}
        for layout in layouts:
            values=np.array([np.mean([r['metrics']['mean_NLL']for r in selected if r['seed']==seed and r['layout']==layout])for seed in seeds]);delta=values-baseline;half=2.7764451051977987*delta.std(ddof=1)/math.sqrt(5)
            result['summaries'][split]['layouts'][layout]={'seed_NLL':values.tolist(),'mean_PPL_ratio':math.exp(float(delta.mean())),
                'MC95_t_PPL_ratio_interval':[math.exp(float(delta.mean()-half)),math.exp(float(delta.mean()+half))],
                'frozen_2pct_gate_pass':bool(delta.mean()+half<=math.log(1.02)),'seed_count':5}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summaries'],indent=2),flush=True)

if __name__=='__main__':main()
