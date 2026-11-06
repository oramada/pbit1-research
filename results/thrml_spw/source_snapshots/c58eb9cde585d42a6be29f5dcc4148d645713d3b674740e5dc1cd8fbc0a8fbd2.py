"""Execute the frozen, complete DistilBERT cross-model evaluation on GB10."""
from pathlib import Path
import argparse,json,hashlib,math,time
import numpy as np
from .prepare_v1 import OUT,sha
from . import causal_v1 as sampler
from .weight_survey_v1 import mapped
from .benchmark_v1 import atomic_json


def observations(logits,labels,reference=None):
    z=logits.astype(np.float64);s=z-z.max(1,keepdims=True);logp=s-np.log(np.exp(s).sum(1,keepdims=True));prob=np.exp(logp)
    ce=-logp[np.arange(len(labels)),labels];correct=(z.argmax(1)==labels).astype(np.int8);truth=np.eye(2)[labels]
    result={'logits':logits,'labels':labels,'CE':ce,'correct':correct,'Brier':np.sum((prob-truth)**2,axis=1)}
    if reference is not None:
        ref=reference.astype(np.float64);q=ref-ref.max(1,keepdims=True);q-=np.log(np.exp(q).sum(1,keepdims=True))
        result['reference_to_variant_KL']=np.sum(np.exp(q)*(q-logp),axis=1)
    return result


def check_plan():
    file=OUT/'distilbert_eval_plan_v1.json';p=json.loads(file.read_text())
    assert sha(p['arrays'])==p['arrays_sha256']and sha(p['layout'])==p['layout_sha256']
    assert sha(p['model_preparation'])==p['model_preparation_sha256']
    prep=json.loads(Path(p['model_preparation']).read_text())
    for filename,digest in prep['files'].items():assert sha(filename)==digest
    assert p['samples']==32768 and p['unique_parameters']==66955010
    assert len(p['cases'])==12 and len(p['preferences'])==40
    return file,p,prep


def gpu_busy():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        if b'research.thrml_spw.spark_depth.benchmark_v1'in args:return True
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();plan_file,p,prep=check_plan()
    if args.check:
        # Loss fixture: uniform binary logits have CE=ln2; a correct confident
        # prediction must have lower CE, correct accuracy and zero self-KL.
        labels=np.array([0,1],np.int64);z=np.array([[2.,-2.],[-2.,2.]],np.float32)
        a=observations(np.zeros((2,2),np.float32),labels);b=observations(z,labels,z)
        np.testing.assert_allclose(a['CE'],np.log(2));assert np.all(b['CE']<a['CE'])and np.all(b['correct']==1)and np.all(b['reference_to_variant_KL']==0)
        print(json.dumps({'status':'plan_assets_and_binary_probability_metrics_checked_no_GPU','reviews':p['reviews'],'cases':12,'GPU_busy':gpu_busy()}));return
    if gpu_busy():raise RuntimeError('The GPT2 benchmark is still running. Keep the GPU lane serial.')
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert'GB10'in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(min(1.,p['GPU_allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'distilbert_benchmark_v1';root.mkdir(exist_ok=True)
    protocol={'plan_sha256':sha(plan_file),'source_sha256':sha(__file__),'sampler_sha256':sha(sampler.__file__),'mapping_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),'plan':p,
        'runtime':{'torch':torch.__version__,'numpy':np.__version__,'GPU':torch.cuda.get_device_name()}}
    if(root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:atomic_json(root/'protocol.json',protocol)
    if(root/'results.json').exists():return
    assert not(root/'failure.json').exists(),'Existing failure requires a new experiment version'
    model=AutoModelForSequenceClassification.from_pretrained(prep['checkpoint'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    parameters=dict(model.named_parameters());original={n:v.detach().cpu().clone()for n,v in parameters.items()};assert sum(v.numel()for v in parameters.values())==66955010 and len(parameters)==104
    with np.load(p['arrays'],allow_pickle=False)as d:ids=d['input_ids'].copy();masks=d['attention_mask'].copy();labels=d['labels'].copy()
    started=time.monotonic();case_results=[]
    try:
        for case in p['cases']:
            folder=root/case['name'];folder.mkdir(exist_ok=True);point=hashlib.sha256();rows=[]
            with torch.no_grad():
                for index,(name,param)in enumerate(parameters.items()):
                    orientation='storage_flat'if case['kind']=='matched_flat'else p['preferences'].get(name,'storage_flat')
                    if case['kind']=='original':value=original[name]
                    elif case['kind']=='mean':
                        w=original[name].float().numpy();o=w.T.copy()if orientation=='transposed_matrix'and w.ndim==2 else w
                        *_,mu,var=mapped(o,p['group_size']);mu=mu.reshape(o.shape);mu=mu.T.copy()if orientation=='transposed_matrix'and w.ndim==2 else mu
                        value=torch.from_numpy(mu.astype(np.float32)).to(torch.bfloat16)
                    else:
                        sampler.SEED=case['seed'];value,record=sampler.sample_parameter(original[name],index,orientation,p['group_size'],p['samples'],torch);record.update(name=name,shape=list(original[name].shape));rows.append(record)
                    point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes());param.copy_(value.cuda())
            point_sha=point.hexdigest();metadata={'point_sha256':point_sha,'parameters':rows};meta=folder/'parameters.json'
            if meta.exists():assert json.loads(meta.read_text())==metadata
            else:atomic_json(meta,metadata)
            sums={'CE':0.,'correct':0.,'Brier':0.,'reference_to_variant_KL':0.};count=0;peak=0
            for index,start in enumerate(range(0,25000,p['batch_size'])):
                stop=min(start+p['batch_size'],25000);file=folder/f'batch{index:04d}.npz';receipt=folder/f'batch{index:04d}.json'
                ih=sampler.digest(ids[start:stop]);mh=sampler.digest(masks[start:stop])
                if receipt.exists():
                    r=json.loads(receipt.read_text());assert r['point_sha256']==point_sha and r['input_sha256']==ih and r['mask_sha256']==mh and sha(file)==r['arrays_sha256']
                    with np.load(file,allow_pickle=False)as d:obs={key:d[key].copy()for key in d.files}
                else:
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():output=model(input_ids=torch.from_numpy(ids[start:stop]).cuda(),attention_mask=torch.from_numpy(masks[start:stop]).cuda())
                    logits=output.logits.float().cpu().numpy()
                    if case['kind']=='original':reference=logits
                    else:
                        with np.load(root/'BF16'/f'batch{index:04d}.npz',allow_pickle=False)as d:reference=d['logits'].copy()
                    obs=observations(logits,labels[start:stop],reference)
                    temp=file.with_suffix('.tmp')
                    with temp.open('wb')as f:np.savez_compressed(f,**obs)
                    temp.replace(file);r={'start':start,'stop':stop,'point_sha256':point_sha,'input_sha256':ih,'mask_sha256':mh,'arrays_sha256':sha(file),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()};atomic_json(receipt,r);del output,logits
                assert np.array_equal(obs['labels'],labels[start:stop])
                for key in sums:assert np.isfinite(obs[key]).all();sums[key]+=float(np.sum(obs[key]))
                count+=stop-start;peak=max(peak,r['GPU_peak_allocated_bytes'])
                atomic_json(root/'progress.json',{'case':case['name'],'batch':index,'completed_reviews':count,'total_reviews':25000,'elapsed_seconds':time.monotonic()-started})
            assert count==25000
            summary={'case':case,'reviews':count,'means':{key:value/count for key,value in sums.items()},'point_sha256':point_sha,'GPU_peak_allocated_bytes':peak};atomic_json(folder/'results.json',summary);case_results.append(summary)
            print(json.dumps({'completed':case['name'],'means':summary['means']}),flush=True)
        base=case_results[0]['means'];stats={}
        for kind in ['selected','matched_flat']:
            chosen=[r['means']for r in case_results if r['case']['kind']==kind];assert len(chosen)==5
            loss=np.array([r['CE']-base['CE']for r in chosen]);decline=np.array([base['correct']-r['correct']for r in chosen]);t=2.7764451051977987
            upper_loss=float(loss.mean()+t*loss.std(ddof=1)/math.sqrt(5));upper_decline=float(decline.mean()+t*decline.std(ddof=1)/math.sqrt(5))
            stats[kind]={'mean_accuracy':float(np.mean([r['correct']for r in chosen])),'accuracy_decline_upper_MC95':upper_decline,'accuracy_pass':upper_decline<=.02,
                'categorical_perplexity_ratio':float(np.exp(loss.mean())),'categorical_perplexity_ratio_upper_MC95':math.exp(upper_loss),'probability_pass':upper_loss<=math.log(1.02),
                'joint_pass':upper_decline<=.02 and upper_loss<=math.log(1.02)}
        atomic_json(root/'results.json',{'status':'complete_cross_model_cases_pending_independent_audit','protocol_sha256':sha(root/'protocol.json'),'cases':case_results,'statistics':stats,'scope':p['scope']})
    except Exception as error:
        atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise

if __name__=='__main__':main()
