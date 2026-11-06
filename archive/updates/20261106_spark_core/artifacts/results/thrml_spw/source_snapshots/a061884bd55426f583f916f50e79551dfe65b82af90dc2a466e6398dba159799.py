"""Separate the two uses of tied embeddings with causal forward overrides."""
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from . import causal_v1 as sampler
from .trace_v1 import score

def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'tied_head_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    names=['BF16','flat','restore_input_use_only','restore_output_use_only','restore_both_uses','only_embedding_reoriented','only_other_matrices_reoriented','all_reoriented']
    protocol={'source_sha256':sha(__file__),'sampler_sha256':sha(sampler.__file__),'seed':sampler.SEED,'conditions':names,'rows':list(range(8)),'input_length':129,
        'scope':'Training-only causal diagnostics. Input/output restoration uses forward hooks to separate uses of one tied parameter; they are interventions, not realizable all-K1 encoding proposals. Reoriented proposals retain124439808pbits,T8192,128weightgroups. R1 idealindependentbinomial,notnativeTHRML.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    params=dict(model.named_parameters());original={k:p.detach().cpu().clone()for k,p in params.items()}
    flat,_=sampler.build_point(original,'storage_flat',128,torch);transposed,_=sampler.build_point(original,'transposed_matrix',128,torch);embedding_original=original['transformer.wte.weight'].cuda()
    with np.load(OUT/'training_diagnostic_windows.npz')as f:windows=f['input_ids'][:8,:129].copy()
    baselines={};records=[];summaries={}
    for name in names:
        with torch.no_grad():
            for key,p in params.items():
                value=original[key]if name=='BF16'else transposed[key]if name=='all_reoriented'or(name=='only_embedding_reoriented'and key=='transformer.wte.weight')or(name=='only_other_matrices_reoriented'and key!='transformer.wte.weight')else flat[key]
                p.copy_(value.cuda())
        handles=[]
        if name in ['restore_input_use_only','restore_both_uses']:handles.append(model.transformer.wte.register_forward_hook(lambda module,args,out:F.embedding(args[0],embedding_original)))
        if name in ['restore_output_use_only','restore_both_uses']:handles.append(model.lm_head.register_forward_hook(lambda module,args,out:F.linear(args[0],embedding_original)))
        try:
            rows=[]
            for row,ids in enumerate(windows):
                with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False)
                logits=out.logits.detach().float().cpu().numpy()
                if name=='BF16':baselines[row]=logits.copy()
                metrics,nll=score(logits,baselines[row],ids[None]);r={'condition':name,'row':row,'metrics':metrics,'logits_sha256':sampler.digest(logits),'input_sha256':sampler.digest(ids)};rows.append(r);records.append(r)
                with(root/f'{name}_row{row}_NLL.npy').open('xb')as f:np.save(f,nll)
                (root/f'{name}_row{row}.json').write_text(json.dumps(r,indent=2)+'\n');(root/'progress.json').write_text(json.dumps({'condition':name,'row':row,'elapsed_seconds':time.monotonic()-begin})+'\n')
                del out,logits
            mean=float(np.mean([r['metrics']['mean_NLL']for r in rows]));summaries[name]={'NLL':mean,'PPL':math.exp(mean)};print(json.dumps({'condition':name,**summaries[name]}),flush=True)
        finally:
            for h in handles:h.remove()
    (root/'results.json').write_text(json.dumps({'status':'completed_training_diagnostic_tied_matrix_role_decomposition','scope':protocol['scope'],'protocol_sha256':sha(root/'protocol.json'),'records':records,'summaries':summaries,'elapsed_seconds':time.monotonic()-begin},indent=2)+'\n')

if __name__=='__main__':main()
