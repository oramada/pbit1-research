"""Reconstruct each DistilBERT point and replay three fixed BF16 CUDA batches."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from .prepare_v1 import OUT,sha
from .weight_survey_v1 import mapped
from . import causal_v1 as sampler

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case');args=parser.parse_args()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=path.read_bytes().split(b'\0')
        except OSError:continue
        if b'research.thrml_spw.spark_depth.distilbert_benchmark_v1'in cmd or b'research.thrml_spw.spark_depth.benchmark_v1'in cmd:raise RuntimeError('Wait for active benchmark GPU execution')
    import torch
    from transformers import AutoModelForSequenceClassification
    from .distilbert_benchmark_v1 import observations
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'distilbert_benchmark_v1';protocol=json.loads((root/'protocol.json').read_text());p=protocol['plan'];prep=json.loads(Path(p['model_preparation']).read_text())
    cases=[c for c in p['cases']if c['name']==args.case]if args.case else p['cases'];assert cases
    if not args.case:assert(root/'results.json').exists()
    for file,h in prep['files'].items():assert sha(file)==h
    with np.load(p['arrays'])as f:ids=f['input_ids'].copy();mask=f['attention_mask'].copy();labels=f['labels'].copy()
    dest=OUT/'distilbert_replay_v1';dest.mkdir(exist_ok=True);results=[]
    for case in cases:
        folder=root/case['name'];assert(folder/'results.json').exists();saved=json.loads((folder/'parameters.json').read_text())
        model=AutoModelForSequenceClassification.from_pretrained(prep['checkpoint'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
        parameters=dict(model.named_parameters());assert len(parameters)==104 and sum(p.numel()for p in parameters.values())==66955010
        point=hashlib.sha256();replayed=0
        with torch.no_grad():
            for i,(name,p_)in enumerate(parameters.items()):
                raw=p_.detach().clone();layout='storage_flat'if case['kind']=='matched_flat'else p['preferences'].get(name,'storage_flat')
                if case['kind']=='original':value=raw
                elif case['kind']=='mean':
                    w=raw.float().numpy();a=w.T.copy()if layout=='transposed_matrix'and w.ndim==2 else w
                    *_,mean,var=mapped(a,128);mean=mean.reshape(a.shape);mean=mean.T.copy()if layout=='transposed_matrix'and w.ndim==2 else mean;value=torch.from_numpy(mean.astype(np.float32)).to(torch.bfloat16)
                else:
                    sampler.SEED=case['seed'];value,r=sampler.sample_parameter(raw,i,layout,128,32768,torch)
                    assert r['counts_uint16_sha256']==saved['parameters'][i]['counts_uint16_sha256']and r['BF16_sha256']==saved['parameters'][i]['BF16_sha256'];replayed+=1
                point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes());p_.copy_(value)
        assert point.hexdigest()==saved['point_sha256'];model.cuda();batch_indices=[0,391,781]
        for index in batch_indices:
            start=index*32;stop=min(start+32,25000)
            with torch.inference_mode():output=model(input_ids=torch.from_numpy(ids[start:stop]).cuda(),attention_mask=torch.from_numpy(mask[start:stop]).cuda())
            logits=output.logits.float().cpu().numpy()
            with np.load(folder/f'batch{index:04d}.npz')as data:
                np.testing.assert_array_equal(logits,data['logits'])
                with np.load(root/'BF16'/f'batch{index:04d}.npz')as base:obs=observations(logits,labels[start:stop],base['logits'])
                for key,value in obs.items():np.testing.assert_array_equal(value,data[key])
        result={'case':case,'point_sha256':point.hexdigest(),'count_vectors_replayed':replayed,'batches':batch_indices,'source_sha256':sha(__file__)}
        (dest/(case['name']+'.json')).write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        (dest/'progress.json').write_text(json.dumps({'completed':len(results),'total':len(cases),'last':case['name']})+'\n');print(case['name'],flush=True);del model
    if not args.case:(dest/'results.json').write_text(json.dumps({'status':'all12points_and36_CUDA_batches_replayed_exactly','source_sha256':sha(__file__),'cases':results,'scope':'Sameimplementationexactreexecution withfreshmodelinstances; independentprobabilityarithmetic audit separate.'},indent=2)+'\n')

if __name__=='__main__':main()
