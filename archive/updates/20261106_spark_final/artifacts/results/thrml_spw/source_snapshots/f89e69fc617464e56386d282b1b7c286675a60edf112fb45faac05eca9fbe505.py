"""Pinned original LFM3 full-checkpoint CPU-master/leaf-CUDA preflight."""
from pathlib import Path
import argparse,hashlib,json,sys,time
from collections import Counter
import numpy as np
from .prepare_v1 import OUT,sha

ROOT=Path('data/thrml_spw/lfm3_1p7b/checkpoint')
REV='ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7'

def prepare():
    from safetensors import safe_open
    files={}
    for name in ['model.safetensors','config.json','configuration_lfm3.py','modeling_lfm3.py','lfm3_conv.py','tokenizer.json','tokenizer_config.json']:
        file=ROOT/name;metadata=(ROOT/'.cache/huggingface/download'/(name+'.metadata')).read_text().splitlines();assert metadata[0]==REV
        digest=sha(file)
        if len(metadata[1])==64:assert digest==metadata[1]
        else:assert hashlib.sha1(b'blob '+str(file.stat().st_size).encode()+b'\0'+file.read_bytes()).hexdigest()==metadata[1]
        files[name]={'sha256':digest,'bytes':file.stat().st_size}
    with safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as f:
        tensors={name:list(f.get_slice(name).get_shape())for name in f.keys()}
    assert len(tensors)==286 and sum(np.prod(v)for v in tensors.values())==1728801536
    old=json.loads(Path('results/thrml_spw/lfm3_1p7b/meta_inventory.json').read_text());assert {r['name']:r['shape']for r in old['parameters']}==tensors
    path=OUT/'lfm3_full_preparation_v1.json';payload={'status':'all_checkpoint_hashes_and_286_shapes_verified','revision':REV,'files':files,'tensors':tensors,'source_sha256':sha(__file__),'scope':'Originalpretrainedcheckpoint,notearlierfine-tunedprobabilityprogram.'}
    if path.exists():assert json.loads(path.read_text())==payload
    else:path.write_text(json.dumps(payload,indent=2)+'\n')
    return path,payload

def busy():
    names={b'research.thrml_spw.spark_depth.benchmark_v1',b'research.thrml_spw.spark_depth.distilbert_benchmark_v1',b'research.thrml_spw.spark_depth.replay_distilbert_v1',b'research.thrml_spw.spark_depth.run_benchmark_replays_v1'}
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=set(file.read_bytes().split(b'\0'))
        except OSError:continue
        if names&args:return True
    return False

def point_hash(model):
    h=hashlib.sha256();count=0
    for name,p in model.named_parameters():
        assert p.device.type=='cpu';h.update(name.encode());h.update(p.detach().view(__import__('torch').uint16).numpy().tobytes());count+=p.numel()
    assert count==1728801536;return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();prepared,meta=prepare()
    if args.prepare_only:print(json.dumps({'status':meta['status'],'GPU_busy':busy()}));return
    if busy():raise RuntimeError('Keep GPU lane serial; another research job is active')
    import torch
    from transformers import AutoModelForCausalLM
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'lfm3_full_preflight_v1';root.mkdir(exist_ok=False)
    protocol={'source_sha256':sha(__file__),'prepared_sha256':sha(prepared),'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),'cases':[[1,17],[1,65]],
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
