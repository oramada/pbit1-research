"""Five-seed cumulative prefix/suffix block substitutions on training inputs."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .counter_sampler_v3 import sample_parameter
from .cpu_vector_validation_v1 import token_losses
from .benchmark_v1 import atomic_json

SEEDS=list(range(2026112801,2026112806))
CONDITIONS=['prefix3','prefix6','prefix9','suffix3','suffix6','suffix9','all_blocks','all_parameters']


def stochastic(name,condition):
    if condition=='all_parameters':return True
    if not name.startswith('transformer.h.'):return False
    layer=int(name.split('.')[2])
    if condition=='all_blocks':return True
    if condition.startswith('prefix'):return layer<int(condition[6:])
    if condition.startswith('suffix'):return layer>=12-int(condition[6:])
    raise ValueError(condition)


def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    root=OUT/'gpt2_cumulative_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    prefs={x['weight']:x['preferred_by_local_variance']for x in json.loads((OUT/'activation_variance_v1/results.json').read_text())['operators']}
    protocol={'source_sha256':sha(__file__),'sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v3.py')),
      'data_sha256':sha(OUT/'training_diagnostic_windows.npz'),'seeds':SEEDS,'conditions':CONDITIONS,'rows':list(range(8)),'tokens':129,'T':32768,'group':128,'preferences':prefs,
      'scope':'CPU FP32 neural arithmetic with BF16-rounded original/sampled weights, 1024 training targets. Prefix/suffix curves use 3,6,9 and12 stochastic blocks with all nonblock parameters deterministic. Only all_parameters converts every learned scalar. These are five-seed causal diagnostics, not new GPU heldout quality tests.'}
    atomic_json(root/'protocol.json',protocol)
    try:
        model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval()
        params=dict(model.named_parameters());original={n:p.detach().to(torch.bfloat16).clone()for n,p in params.items()}
        with np.load(OUT/'training_diagnostic_windows.npz')as f:inputs=f['input_ids'][:8,:129].copy()
        capture={}
        handles=[layer.register_forward_hook(lambda m,a,o,i=i:capture.__setitem__(i,o[0].detach().numpy().copy()))for i,layer in enumerate(model.transformer.h)]
        baseline={};summaries=[]
        for seed in [None]+SEEDS:
            draws={}
            if seed is not None:
                records=[]
                for index,(name,weight)in enumerate(original.items()):
                    value,record=sample_parameter(weight,index,prefs.get(name,'storage_flat'),128,32768,torch,seed)
                    draws[name]=value;records.append({'name':name,**record})
                atomic_json(root/f'seed{seed}_draws.json',records)
            for condition in ['baseline']if seed is None else CONDITIONS:
                label=condition if seed is None else f'{condition}_seed{seed}';folder=root/label;folder.mkdir();point=hashlib.sha256();converted=0
                with torch.no_grad():
                    for name,param in params.items():
                        use=seed is not None and stochastic(name,condition)
                        value=draws[name]if use else original[name]
                        point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes());param.copy_(value.float());converted+=param.numel()if use else 0
                losses=[];errors=np.zeros((12,768));signal=np.zeros_like(errors)
                for row,ids in enumerate(inputs):
                    capture.clear()
                    with torch.inference_mode():out=model(input_ids=torch.from_numpy(ids[None]),use_cache=False)
                    loss=token_losses(out.logits.numpy(),ids);losses.extend(loss.tolist());del out
                    if seed is None:baseline[row]={i:a.copy()for i,a in capture.items()}
                    for layer in range(12):
                        a=capture[layer].reshape(-1,768).astype(np.float64);b=baseline[row][layer].reshape(-1,768).astype(np.float64);errors[layer]+=((a-b)**2).sum(0);signal[layer]+=(b*b).sum(0)
                    with(folder/f'row{row}.npz').open('xb')as f:np.savez_compressed(f,token_NLL=loss,input_ids=ids)
                if condition.startswith('suffix'):assert np.count_nonzero(errors[:12-int(condition[6:])])==0
                with(folder/'channels.npz').open('xb')as f:np.savez_compressed(f,error_SSE=errors,signal_SSE=signal)
                record={'seed':seed,'condition':condition,'NLL':float(np.mean(losses)),'targets':len(losses),'converted_parameters':converted,'point_sha256':point.hexdigest()}
                summaries.append(record);atomic_json(folder/'results.json',record);atomic_json(root/'progress.json',{'condition':condition,'seed':seed,'elapsed_seconds':time.monotonic()-begin});print(json.dumps(record),flush=True)
        assert not torch.cuda.is_initialized()
        atomic_json(root/'results.json',{'status':'all41_cumulative_CPU_training_conditions_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'cases':summaries,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
    except Exception as error:
        atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise

if __name__=='__main__':main()
