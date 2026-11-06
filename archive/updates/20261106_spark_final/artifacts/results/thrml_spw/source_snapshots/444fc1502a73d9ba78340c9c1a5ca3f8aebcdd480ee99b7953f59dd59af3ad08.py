"""Full LFM3 offload preflight including the planned 1,024-token context."""
from pathlib import Path
import argparse,json,sys,time
from collections import Counter
import numpy as np
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT,prepare,busy,point_hash

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();prepared,meta=prepare()
    if args.prepare_only:print(json.dumps({'status':meta['status'],'GPU_busy':busy()}));return
    if busy():raise RuntimeError('Keep GPU lane serial; another research job is active')
    import torch
    from transformers import AutoModelForCausalLM
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'lfm3_full_preflight_v2';root.mkdir(exist_ok=False)
    protocol={'source_sha256':sha(__file__),'prepared_sha256':sha(prepared),'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),'cases':[[1,17],[1,65],[1,1024]],
        'GPU_cap_bytes':2*1024**3,'use_cache':False,'wide_conv_kernel':False,'scope':'Fulloriginalcheckpoint,syntheticIDs only. Tests finiteforwards,modulecoverage,weightimmutabilityandGPUallocation; nottaskquality.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');begin=time.monotonic()
    try:
        model=AutoModelForCausalLM.from_pretrained(ROOT,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
        sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
        expected=Counter(n for n,m in model.named_modules()if any(True for _ in m.parameters(recurse=False)));assert sum(expected.values())==287
        before=point_hash(model);assert model.lm_head.weight is model.model.embed_tokens.weight
        results=[]
        with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
            for index,(batch,length)in enumerate(protocol['cases']):
                offloader.calls.clear();torch.cuda.reset_peak_memory_stats();ids=(torch.arange(batch*length).reshape(batch,length)*7919+17)%128000
                output=model(input_ids=ids.cuda(),use_cache=False);logits=output.logits.detach().cpu().float().numpy()
                assert logits.shape==(batch,length,128000)and np.isfinite(logits).all()
                assert Counter(offloader.calls)==expected and offloader.active_bytes==0
                assert model.lm_head.weight is model.model.embed_tokens.weight and point_hash(model)==before
                peak=torch.cuda.max_memory_allocated();assert peak<=2*1024**3
                file=root/f'case{index}_logits.npz'
                with file.open('xb')as f:np.savez_compressed(f,input_ids=ids.numpy(),logits=logits)
                result={'batch':batch,'length':length,'logits_shape':list(logits.shape),'file_sha256':sha(file),'weight_point_sha256':before,'Torch_GPU_peak_allocated_bytes':peak,
                    'peak_materialized_parameter_bytes':offloader.peak_active_bytes,'parameterized_module_calls':len(offloader.calls)}
                results.append(result);(root/'progress.json').write_text(json.dumps({'completed_cases':len(results),'elapsed_seconds':time.monotonic()-begin})+'\n');print(json.dumps(result),flush=True);del output,logits
        (root/'results.json').write_text(json.dumps({'status':'full_LFM3_original_checkpoint_offload_preflight_passed','protocol_sha256':sha(root/'protocol.json'),'cases':results,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']},indent=2)+'\n')
    except Exception as error:
        (root/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2)+'\n');raise

if __name__=='__main__':main()
