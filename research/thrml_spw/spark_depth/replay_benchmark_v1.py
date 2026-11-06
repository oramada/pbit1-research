"""Deterministic point replay; optional local neural replay after GPU job ends."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .profile_calibration_v1 import policy
from .weight_survey_v1 import mapped
from . import causal_v1 as sampler

def main():
    args=argparse.ArgumentParser();args.add_argument('--case',required=True);args.add_argument('--with-forward',action='store_true');args=args.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    root=OUT/'benchmark_v1';p=json.loads((root/'protocol.json').read_text());case=next(c for c in p['cases']if c['name']==args.case)
    folder=root/case['name'];assert(folder/'results.json').exists(),'Wait for case closure'
    prior=json.loads((folder/'parameters.json').read_text());assert sha(CHECKPOINT/'model.safetensors')==p['checkpoint_sha256']
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
    params=dict(model.named_parameters());old=json.loads(Path('results/thrml_spw/broad_verification_20261105/remote_metadata/additional_parameter_packages_v1/gpt2/package.json').read_text())
    assert list(params)==[r['name']for r in old['canonical_parameters']];point=hashlib.sha256();replayed=0
    with torch.no_grad():
        for index,((name,param),original_record)in enumerate(zip(params.items(),old['canonical_parameters'])):
            raw=param.detach().clone();assert sampler.digest(raw.view(torch.uint16).numpy())==original_record['original_BF16_raw_sha256']
            orientation,T=policy(name,p['selected_profile'],p['preferences'])
            if case['kind']=='matched_flat':orientation='storage_flat'
            if case['kind']=='original':value=raw
            elif case['kind']=='mean':
                w=raw.float().numpy();o=w.T.copy()if orientation=='transposed_matrix'and w.ndim==2 else w
                *_,mu,var=mapped(o,128);mu=mu.reshape(o.shape);mu=mu.T.copy()if orientation=='transposed_matrix'and w.ndim==2 else mu;value=torch.from_numpy(mu.astype(np.float32)).to(torch.bfloat16)
            else:
                sampler.SEED=case['seed'];value,record=sampler.sample_parameter(raw,index,orientation,128,T,torch)
                assert record['counts_uint16_sha256']==prior['records'][index]['counts_uint16_sha256']
                assert record['BF16_sha256']==prior['records'][index]['BF16_sha256'];replayed+=1
            point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes());param.copy_(value)
    assert point.hexdigest()==prior['point_sha256'];forward=[]
    if args.with_forward:
        from research.thrml_spw.broad.r1_model_v1 import observations
        assert'GB10'in torch.cuda.get_device_name();torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model.cuda();assert model.lm_head.weight is model.transformer.wte.weight
        with np.load(p['data']['tokens'])as d:tokens=d['tokens'].copy();offsets=d['offsets'].copy()
        layout=json.loads(Path(p['data']['layout']).read_text())
        for index in [0,293,585]:
            w=layout['windows'][index];start=int(offsets[w['article']]);ids=tokens[start+w['context_start']:start+w['context_stop']][None]
            with torch.inference_mode():output=model(input_ids=torch.from_numpy(ids).cuda(),use_cache=False)
            logits=output.logits.float().cpu().numpy();receipt=json.loads((folder/f'window{index:04d}.json').read_text());assert sampler.digest(logits)==receipt['logits_sha256']
            obs=observations({'task':'gpt2',**w},ids,None,logits)
            with np.load(folder/f'window{index:04d}.npz')as d:
                for key,value in obs.items():np.testing.assert_array_equal(value,d[key])
            forward.append(index)
    else:assert not torch.cuda.is_initialized()
    dest=OUT/'benchmark_replay_v1';dest.mkdir(exist_ok=True)
    name=args.case+('_neural'if args.with_forward else'_weights')+'.json'
    result={'status':'passed_deterministic_point_replay','case':case,'source_sha256':sha(__file__),'point_sha256':point.hexdigest(),'count_vectors_replayed':replayed,'neural_windows_replayed':forward,
        'scope':'Samefrozenimplementation reexecution, notanindependentneuralengine. CPUweights-onlymode allocatesnoGPU. Forwardmode validatesexactGB10logitshashes andsavedobservations onthreeprespecifiedwindows.'}
    (dest/name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
