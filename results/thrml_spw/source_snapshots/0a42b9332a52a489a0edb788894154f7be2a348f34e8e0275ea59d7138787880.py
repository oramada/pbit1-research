"""Prespecified isolated and rescued LFM3 layers on training inputs only."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,sys,time
import numpy as np
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT,point_hash
from .counter_sampler_v3 import sample_parameter
from .benchmark_v1 import atomic_json

SEED=2026111801

def belongs(name,scope):
    if scope=='all':return True
    if scope.startswith('layer:'):return name.startswith('model.layers.'+scope.split(':')[1]+'.')
    if scope=='embedding':return name=='model.embed_tokens.weight'
    if scope=='norm':return 'norm.weight'in name
    if scope=='attention':return '.self_attn.'in name and 'norm.weight'not in name
    if scope=='convolution':return '.conv.'in name
    if scope=='feed_forward':return '.feed_forward.'in name
    raise ValueError(scope)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    source=OUT/'lfm3_calibration_v2/results.json';policies=json.loads(source.read_text())['policies'];audit=OUT/'lfm3_calibration_v3_audit.json';assert json.loads(audit.read_text())['result_sha256']==sha(source)
    cases=[{'name':'BF16','mode':'isolated','scope':'none'},{'name':'all_K1','mode':'isolated','scope':'all'}]
    cases += [{'name':f'{mode}_layer{i:02d}','mode':mode,'scope':f'layer:{i}'}for i in range(24)for mode in ['isolated','rescued']]
    cases += [{'name':'rescued_'+group,'mode':'rescued','scope':group}for group in ['embedding','norm','attention','convolution','feed_forward']]
    protocol={'source_sha256':sha(__file__),'calibration_sha256':sha(source),'audit_sha256':sha(audit),'sampler_sha256':sha(Path(__file__).with_name('counter_sampler_v3.py')),
      'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),'training_data_sha256':sha(OUT/'lfm3_data_v1/training_windows.npz'),
      'cases':cases,'seed':SEED,'samples':32768,'training_rows':list(range(8)),'tokens_per_row':129,
      'scope':'Causal fixed-weight-draw interventions on training data. Isolated: selected subset stochastic, rest original BF16. Rescued: subset original, rest stochastic. Mixed cases intentionally relax all-parameter K1 to diagnose causality; not deliverable compression results. One seed is mechanistic, not statistical generality.'}
    if args.check:print(json.dumps(protocol));return
    import fcntl
    lock=(OUT/'gpu_research.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name();torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'lfm3_causal_depth_v1';root.mkdir(exist_ok=True)
    if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:atomic_json(root/'protocol.json',protocol)
    assert not(root/'failure.json').exists()
    if(root/'results.json').exists():return
    begin=time.monotonic()
    try:
        model=AutoModelForCausalLM.from_pretrained(ROOT,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
        sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
        params=dict(model.named_parameters());assert list(params)==[p['name']for p in policies]
        assert point_hash(model)==json.loads((OUT/'lfm3_full_preflight_v2_audit.json').read_text())['weight_point_sha256']
        expected=Counter(n for n,m in model.named_modules()if any(True for _ in m.parameters(recurse=False)))
        draws={};records=[]
        with safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as weights:
            for index,p in enumerate(policies):
                raw=weights.get_tensor(p['name']);draw,record=sample_parameter(raw,index,p['selected_layout'],p['selected_group'],32768,torch,SEED);draws[p['name']]=draw;record['name']=p['name'];records.append(record)
                atomic_json(root/'progress.json',{'stage':'draw','parameter':index,'elapsed_seconds':time.monotonic()-begin})
        # The draw must match the same seed in the complete training study, tensor by tensor.
        prior=json.loads((OUT/f'lfm3_training_depth_v1/selected_seed{SEED}/parameters.json').read_text())
        assert records==prior['records'];atomic_json(root/'draw_records.json',{'records':records})
        with np.load(OUT/'lfm3_data_v1/training_windows.npz')as f:inputs=f['input_ids'][:8,:129].copy()
        capture={};handles=[]
        for layer,m in enumerate(model.model.layers):handles.append(m.register_forward_hook(lambda m,a,o,layer=layer:capture.__setitem__(layer,o.detach().cpu().float().numpy())))
        baseline={};summaries=[];installed={name:False for name in params}
        for case in cases:
            folder=root/case['name'];folder.mkdir(exist_ok=True);stochastic={name:False if case['scope']=='none'else belongs(name,case['scope'])!=(case['mode']=='rescued')for name in params}
            with torch.no_grad(),safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as weights:
                for name,p in params.items():
                    if stochastic[name]!=installed[name]:p.copy_(draws[name]if stochastic[name]else weights.get_tensor(name))
            installed=stochastic;point=point_hash(model);losses=[];errors=np.zeros((24,2048));signal=np.zeros_like(errors);peak=0
            with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
                for row,ids in enumerate(inputs):
                    offloader.calls.clear();capture.clear();torch.cuda.reset_peak_memory_stats();out=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False)
                    assert Counter(offloader.calls)==expected and len(capture)==24
                    loss=torch.nn.functional.cross_entropy(out.logits[0,:-1].float(),torch.from_numpy(ids[1:]).cuda(),reduction='none').cpu().double().numpy();assert np.isfinite(loss).all();del out
                    if case['name']=='BF16':baseline[row]={i:a.copy()for i,a in capture.items()}
                    for layer in range(24):
                        b=baseline[row][layer].reshape(-1,2048).astype(np.float64);a=capture[layer].reshape(-1,2048).astype(np.float64);errors[layer]+=((a-b)**2).sum(0);signal[layer]+=(b*b).sum(0)
                    losses.extend(loss.tolist());peak=max(peak,torch.cuda.max_memory_allocated());assert peak<=2*1024**3
                    saved_row=folder/f'row{row:02d}.npz'
                    if saved_row.exists():
                        with np.load(saved_row)as f:np.testing.assert_array_equal(f['token_NLL'],loss);np.testing.assert_array_equal(f['input_ids'],ids)
                    else:
                        with saved_row.open('xb')as f:np.savez_compressed(f,token_NLL=loss,input_ids=ids)
                    atomic_json(root/'progress.json',{'stage':'causal_forward','case':case['name'],'row':row,'elapsed_seconds':time.monotonic()-begin})
            del offloader
            if(folder/'channels.npz').exists():
                with np.load(folder/'channels.npz')as f:np.testing.assert_array_equal(f['error_SSE'],errors);np.testing.assert_array_equal(f['signal_SSE'],signal)
            else:
                with(folder/'channels.npz').open('xb')as f:np.savez_compressed(f,error_SSE=errors,signal_SSE=signal)
            summary={'case':case,'NLL':float(np.mean(losses)),'targets':len(losses),'point_sha256':point,'stochastic_parameters':sum(params[n].numel()for n,s in stochastic.items()if s),'GPU_peak_allocated_bytes':peak,'layer_relative_RMSE':np.sqrt(errors.sum(1)/signal.sum(1)).tolist()}
            atomic_json(folder/'results.json',summary);summaries.append(summary);print(json.dumps(summary),flush=True)
        atomic_json(root/'results.json',{'status':'all55_causal_training_interventions_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'cases':summaries,'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
    except Exception as error:
        atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
