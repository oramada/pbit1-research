"""Full-model K1 LFM3 training-cohort quality and all-layer/channel traces."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,sys,time
import numpy as np
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT,point_hash
from .counter_sampler_v3 import sample_parameter
from .weight_survey_v1 import mapped
from .benchmark_v1 import atomic_json

SEEDS=list(range(2026111801,2026111806))

def mean_parameter(raw,layout,group,torch):
    w=raw.float().numpy();a=w.T.copy()if layout=='transposed_matrix'else w;flat=a.reshape(-1);value=np.empty(flat.size,np.float32);chunk=262144//group*group
    for start in range(0,flat.size,chunk):value[start:start+chunk]=mapped(flat[start:start+chunk],group)[3]
    value=value.reshape(a.shape)
    if layout=='transposed_matrix':value=value.T.copy()
    return torch.from_numpy(value.copy()).to(torch.bfloat16)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    selection=OUT/'lfm3_calibration_v2/results.json';audit=OUT/'lfm3_calibration_v3_audit.json'
    policies=json.loads(selection.read_text())['policies'];assert json.loads(audit.read_text())['result_sha256']==sha(selection)
    verify=json.loads((OUT/'counter_sampler_v3_verification.json').read_text());assert verify['sampler_sha256']==sha(Path(__file__).with_name('counter_sampler_v3.py'))
    root=OUT/'lfm3_training_depth_v1'
    cases=[{'name':'BF16','kind':'original'},{'name':'selected_mean','kind':'mean'}]+[{'name':f'selected_seed{s}','kind':'selected','seed':s}for s in SEEDS]
    protocol={'source_sha256':sha(__file__),'selection_sha256':sha(selection),'selection_audit_sha256':sha(audit),'sampler_sha256':verify['sampler_sha256'],
      'mean_codec_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),'offload_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),
      'training_data_sha256':sha(OUT/'lfm3_data_v1/training_windows.npz'),'cases':cases,'training_rows':list(range(32)),'tokens_per_row':257,
      'parameters':1728801536,'samples':32768,'controller_bits':12,'range_metadata_bits_per_group':64,'state_bits_per_parameter':1,
      'observations_per_reconstruction':1728801536*32768,'statistic':'Five-seed paired deltaNLL on 8192 training targets; t4 95% interval. Calibration diagnostic only; no heldout success claim.',
      'scope':'All learned parameters K1 including norms and convolution filters, tied head counted once. Ideal independent binomial R1; model reconstructed once per seed then reused. CPU-master BF16 arithmetic on shared GB10, 2GiB allocator cap. No native fullmodel THRML or device claim.'}
    if args.check:print(json.dumps(protocol));return
    # Serial lane lock. Cooperating future Spark runners must use this same lock.
    import fcntl
    lock=(OUT/'gpu_research.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert 'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root.mkdir(exist_ok=True)
    if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:atomic_json(root/'protocol.json',protocol)
    assert not(root/'failure.json').exists(),'Preserve failed attempts; use a new version.'
    if(root/'results.json').exists():return
    begin=time.monotonic()
    try:
        model=AutoModelForCausalLM.from_pretrained(ROOT,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
        sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
        params=dict(model.named_parameters());assert list(params)==[p['name']for p in policies]
        assert model.lm_head.weight is model.model.embed_tokens.weight
        expected=Counter(n for n,m in model.named_modules()if any(True for _ in m.parameters(recurse=False)))
        with np.load(OUT/'lfm3_data_v1/training_windows.npz')as f:inputs=f['input_ids'].copy()
        assert inputs.shape==(32,257)
        capture={};handles=[]
        for i,layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(lambda m,a,o,i=i:capture.__setitem__(f'layer{i:02d}',o.detach().cpu().view(torch.uint16).numpy().copy())))
        summaries=[]
        for case in cases:
            folder=root/case['name'];folder.mkdir(exist_ok=True);records=[]
            with safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as weights,torch.no_grad():
                for index,p in enumerate(policies):
                    raw=weights.get_tensor(p['name'])
                    if case['kind']=='original':value=raw
                    elif case['kind']=='mean':value=mean_parameter(raw,p['selected_layout'],p['selected_group'],torch)
                    else:
                        value,record=sample_parameter(raw,index,p['selected_layout'],p['selected_group'],32768,torch,case['seed']);record['name']=p['name'];records.append(record)
                    params[p['name']].copy_(value);del raw,value
                    if index%25==0:atomic_json(root/'progress.json',{'stage':'construct','case':case['name'],'parameter':index,'elapsed_seconds':time.monotonic()-begin})
            point=point_hash(model)
            if case['kind']=='original':assert point==json.loads((OUT/'lfm3_full_preflight_v2_audit.json').read_text())['weight_point_sha256']
            receipt={'point_sha256':point,'records':records}
            if(folder/'parameters.json').exists():assert json.loads((folder/'parameters.json').read_text())==receipt
            else:atomic_json(folder/'parameters.json',receipt)
            losses=[];signals=np.zeros((24,2048));errors=np.zeros_like(signals);max_error=np.zeros_like(signals);dots=np.zeros_like(signals);sample_energy=np.zeros_like(signals);peak=0
            with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
                for row,ids in enumerate(inputs):
                    capture.clear();offloader.calls.clear();torch.cuda.reset_peak_memory_stats()
                    output=model(input_ids=torch.from_numpy(ids[None]).cuda(),use_cache=False)
                    assert Counter(offloader.calls)==expected and offloader.active_bytes==0 and len(capture)==24
                    logits=output.logits[0,:-1].float()
                    loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(ids[1:]).cuda(),reduction='none').cpu().double().numpy()
                    assert loss.shape==(256,) and np.isfinite(loss).all()
                    del logits,output
                    baseline_path=root/'BF16'/f'layer_states{row:02d}.npz'
                    if case['kind']=='original':
                        if baseline_path.exists():
                            with np.load(baseline_path)as existing:
                                for key in capture:np.testing.assert_array_equal(existing[key],capture[key])
                        else:
                            with baseline_path.open('xb')as f:np.savez_compressed(f,**capture)
                    with np.load(baseline_path)as baseline:
                        for layer in range(24):
                            key=f'layer{layer:02d}';b=torch.from_numpy(baseline[key].copy()).view(torch.bfloat16).float().numpy().astype(np.float64).reshape(-1,2048)
                            y=torch.from_numpy(capture[key]).view(torch.bfloat16).float().numpy().astype(np.float64).reshape(-1,2048);e=y-b
                            signals[layer]+=(b*b).sum(0);errors[layer]+=(e*e).sum(0);max_error[layer]=np.maximum(max_error[layer],np.abs(e).max(0));dots[layer]+=(b*y).sum(0);sample_energy[layer]+=(y*y).sum(0)
                    file=folder/f'row{row:02d}.npz'
                    if file.exists():
                        with np.load(file)as old:np.testing.assert_array_equal(old['token_NLL'],loss)
                    else:
                        with file.open('xb')as f:np.savez_compressed(f,token_NLL=loss,input_ids=ids)
                    peak=max(peak,torch.cuda.max_memory_allocated());assert peak<=2*1024**3;losses.extend(loss.tolist())
                    atomic_json(root/'progress.json',{'stage':'neural_forward','case':case['name'],'row':row,'elapsed_seconds':time.monotonic()-begin,'GPU_peak_allocated_bytes':peak})
            with(folder/'channels.npz').open('wb')as f:np.savez_compressed(f,signal_SSE=signals,error_SSE=errors,max_abs_error=max_error,signal_sample_dot=dots,sample_SSE=sample_energy)
            del offloader
            summary={'case':case,'NLL':float(np.mean(losses)),'PPL':math.exp(float(np.mean(losses))),'targets':len(losses),'point_sha256':point,'GPU_peak_allocated_bytes':peak,
              'layer_relative_RMSE':np.sqrt(errors.sum(1)/signals.sum(1)).tolist(),'layer_types':model.config.layer_types}
            atomic_json(folder/'results.json',summary);summaries.append(summary);print(json.dumps(summary),flush=True)
        values=np.array([s['NLL']for s in summaries[2:]])-summaries[0]['NLL'];half=2.7764451051977987*values.std(ddof=1)/np.sqrt(5)
        atomic_json(root/'results.json',{'status':'seven_fullmodel_training_cases_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'cases':summaries,'PPL_ratio':float(np.exp(values.mean())),'MC95_t4_PPL_ratio_interval':np.exp([values.mean()-half,values.mean()+half]).tolist(),'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-begin})
    except Exception as error:
        atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
