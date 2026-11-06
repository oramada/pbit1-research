"""Independent saved-count weight reconstruction and complete neural replay.

No THRML import or random sampling. Every output batch is reproduced and its
raw float32 hash compared; token losses use NumPy rather than producer Torch.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from .r1_io_v1 import CaseStore,sha,array_sha,program,source_guard
from .r1_model_v1 import load_model


def reference(record,counts=None,samples=None,arm=None,original=None):
    p=program(record['program']);codes=p['codes'];n=len(codes);g=np.arange(n)//128
    lo=p['lower'][g].astype(np.float64);span=p['span'][g].astype(np.float64)
    if counts is not None:
        assert counts.dtype==np.uint16 and counts.ndim==2 and counts.shape[1]==n
        assert counts.shape[0] in (1,4,8) and np.all(counts<=samples)
        numerator=np.zeros(n,np.int64)
        for b in range(counts.shape[0]):numerator+=counts[b].astype(np.int64)*(2**b)
        prob=numerator.astype(np.float64)/(samples*((2**counts.shape[0])-1))
    elif arm.startswith('mean'):
        bits=int(arm[4:]);h=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)
        prob=1/(1+np.exp(-2*h.astype(np.float64)))
        if bits!=1:
            endpoint=1/(1+math.exp(8));word=np.rint(np.clip((prob-endpoint)/(1-2*endpoint),0,1)*((2**bits)-1))
            prob=endpoint+(1-2*endpoint)*word/((2**bits)-1)
    elif arm.startswith('int'):
        bits=int(arm[3:]);w=np.asarray(original,np.float32).reshape(-1)
        prob=np.divide(w.astype(np.float64)-lo,span,out=np.full(n,.5),where=span!=0)
        prob=np.rint(np.clip(prob,0,1)*((2**bits)-1))/((2**bits)-1)
    else:raise ValueError('Unknown reference readout')
    return (lo+span*prob).reshape(tuple(p['shape'])).astype(np.float32)


def independent_batches(spec):
    data=spec['data']
    if data['kind']=='synthetic':
        with np.load(data['arrays'],allow_pickle=False) as a:
            yield {'index':0,'task':spec['model_type'],'synthetic':True},a['input_ids'].copy(),a['attention_mask'].copy() if 'attention_mask'in a else None,None
    elif spec['model_type']=='gpt2':
        layout=json.loads(Path(data['layout']).read_text())
        with np.load(data['arrays'],allow_pickle=False) as a:stream=a['tokens'].copy();offset=a['offsets'].copy()
        for i,window in enumerate(layout['windows']):
            article=window['article'];start=int(offset[article])+window['context_start'];stop=int(offset[article])+window['context_stop']
            ids=stream[start:stop].reshape(1,-1)
            yield {'index':i,'task':'gpt2',**window},ids,None,None
    else:
        with np.load(data['arrays'],allow_pickle=False) as a:ids=a['input_ids'].copy();mask=a['attention_mask'].copy();labels=a['labels'].copy()
        for index in range(math.ceil(len(labels)/128)):
            first,last=index*128,min(index*128+128,len(labels))
            yield {'index':index,'task':'distilbert','review_start':first,'review_stop':last},ids[first:last],mask[first:last],labels[first:last]


def independent_observations(record,ids,labels,logits):
    if record.get('synthetic'):return {'logits':logits}
    if record['task']=='distilbert':return {'logits':logits,'labels':labels,'correct':np.equal(logits.argmax(axis=1),labels).astype(np.int8)}
    a=record['target_start']-record['context_start'];b=record['target_stop']-record['context_start']
    losses=[];normalizers=[];selected=[]
    for first in range(a,b,8):
        last=min(first+8,b);z=logits[0,first-1:last-1].astype(np.float64);maximum=np.max(z,axis=1)
        normal=maximum+np.log(np.exp(z-maximum[:,None]).sum(axis=1))
        target=np.array([z[i,int(ids[0,first+i])] for i in range(last-first)])
        losses.extend(normal-target);normalizers.extend(normal);selected.extend(target)
    return {'token_NLL':np.asarray(losses),'log_normalizer':np.asarray(normalizers),
            'target_logit':np.asarray(selected),'target_ids':ids[0,a:b]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--case',required=True)
    args=ap.parse_args();p=json.loads(args.protocol.read_text());source_guard(p)
    case=p['cases'][args.case];spec=p['models'][case['model_key']];root=(Path(p['output'])/'cases'/args.case).resolve()
    result=json.loads((root/'results.json').read_text());assert result['protocol_sha256']==sha(args.protocol)
    assert result['case_spec']==case and result['case']==args.case and result['package_sha256']==spec['package_sha256']
    assert result['status']=='complete_R1_case_produced_pending_independent_neural_replay'
    synthetic=p['status']=='actual_model_synthetic_preflight_no_task_scoring';assert result['synthetic_only']==synthetic
    audit_path=root/'independent_audit.json'
    if audit_path.exists():
        old=json.loads(audit_path.read_text());assert old['results_sha256']==sha(root/'results.json') and old['source_sha256']==sha(__file__)
        return
    contract={'protocol_sha256':sha(args.protocol),'case':args.case,'case_spec':case,'model_spec':spec}
    store=CaseStore(root,contract)
    audit_store=CaseStore(root/'replay',{'results_sha256':sha(root/'results.json'),'source_sha256':sha(__file__)})
    if list((audit_store.root/'attempts').glob('*/failure.json')):raise RuntimeError('Preserved replay failure is terminal')
    attempt=audit_store.new_attempt();begin=time.monotonic()
    try:
        model,parameters,package=load_model(spec);points=[];resources=[]
        for record in package['canonical_parameters']:
            index=record['index'];name=record['name'];assert sha(record['program'])==record['program_sha256']
            if case['mode']=='native':
                closed=store.load(index,'parameter');assert closed is not None and closed['module_calls']==[name]
                path=store.root/closed['output']['path'];assert sha(path)==closed['output']['sha256']
                with np.load(path,allow_pickle=False) as a:
                    assert set(a.files)=={'counts','bits','samples','seed','index'}
                    assert all(int(a[key])==case[key] for key in ['bits','samples','seed']) and int(a['index'])==index
                    counts=a['counts'].copy()
                value=reference(record,counts,case['samples'])
                receipt,=closed['operator_calls'];assert receipt['ok'] and receipt['finite'] and not receipt['reference_weights_supplied']
                assert receipt['counts_sha256']==sha(path) and receipt['program_sha256']==record['program_sha256']
                assert receipt['decoded_FP32_sha256']==array_sha(value)
                assert receipt['readout']=='exactint64positionalnumerator_then_FP64divide_mul_add'
                assert all(receipt[key]==case[key] for key in ['bits','samples','seed']) and receipt['index']==index
                n=record['elements'];k=case['bits'];t=case['samples'];padded=math.ceil(n/65536)*65536;r=receipt['resources']
                assert counts.shape==(k,n) and counts.dtype==np.uint16
                assert r['unique_parameter_scalars']==n and r['logical_pbits']==k*n
                assert r['logical_retained_device_evaluations']==k*n*t and r['scheduled_retained_device_evaluations']==k*padded*t
                assert r['scheduled_warmup_device_evaluations']==k*padded and r['sampling_padding_cells_per_bitplane']==padded-n
                assert r['range_metadata_bits']==64*math.ceil(n/128)
                assert r['finite_control_information_bits']==(12*n if k==1 else k*n)
                assert r['software_sampler_state_slots']==65536 and r['software_int32_count_carry_bytes']==4*65536
                resources.append({'logical_pbits':k*n,'retained_device_evaluations':k*padded*t,
                    'warmup_device_evaluations':k*padded,'logical_retained_device_evaluations':k*n*t})
            elif case['arm']=='original':value=None
            else:value=reference(record,arm=case['arm'],original=parameters[name].detach().float().cpu().numpy())
            if value is not None:
                rounded=torch.from_numpy(value.copy()).to(dtype=torch.bfloat16);parameters[name].data=rounded.to(device='cuda')
            digest=array_sha(parameters[name].detach().cpu().view(torch.uint16).numpy())
            point={'index':index,'name':name,'BF16_sha256':digest};points.append(point)
            assert result['parameter_points'][index]==point
            if case['mode']=='native':assert closed['metrics']=={'name':name,'BF16_sha256':digest}
        assert len(result['parameter_points'])==len(points)
        point_hash=hashlib.sha256(json.dumps(points,sort_keys=True).encode()).hexdigest();assert result['model_point_sha256']==point_hash
        if synthetic:unit_sums=np.zeros(1,np.float64);unit_sizes=np.ones(1,np.int64)
        elif spec['model_type']=='gpt2':
            layout=json.loads(Path(spec['data']['layout']).read_text());unit_sizes=np.asarray([v['target_count'] for v in layout['articles']],np.int64)
            unit_sums=np.zeros(len(unit_sizes),np.float64)
        else:unit_sizes=np.ones(25000,np.int64);unit_sums=np.zeros(25000,np.float64)
        total_batches=0
        for record,ids,mask,labels in independent_batches(spec):
            index=record['index'];original=store.load(index,'evaluation');assert original is not None
            assert original==result['evaluation'][index] and original['metrics']['batch']==record
            assert original['metrics']['model_point_sha256']==point_hash
            assert original['metrics']['input_ids_sha256']==array_sha(ids)
            assert original['metrics']['attention_mask_sha256']==(array_sha(mask) if mask is not None else None)
            verified=audit_store.load(index,'evaluation')
            if verified is None:
                kwargs={'input_ids':torch.as_tensor(ids,device='cuda')}
                if mask is not None:kwargs['attention_mask']=torch.as_tensor(mask,device='cuda')
                if spec['model_type']=='gpt2':kwargs['use_cache']=False
                with torch.no_grad():logits=model(**kwargs).logits.to(dtype=torch.float32).cpu().numpy()
                assert np.isfinite(logits).all() and list(logits.shape)==original['metrics']['logits_shape']
                assert array_sha(logits)==original['metrics']['full_logits_FP32_sha256']
                expected=independent_observations(record,ids,labels,logits)
                with np.load(store.root/original['output']['path'],allow_pickle=False) as a:
                    assert set(a.files)==set(expected)
                    for key,value in expected.items():
                        if key in ['logits','labels','correct','target_ids']:np.testing.assert_array_equal(a[key],value)
                        else:np.testing.assert_allclose(a[key],value,rtol=0,atol=1e-12)
                path=attempt/f'verified{index:05d}.npz'
                with path.open('xb') as f:np.savez_compressed(f,**expected)
                verified={'output':{'path':str(path.relative_to(audit_store.root)),'sha256':sha(path)},
                    'metrics':{'batch':record,'raw_logits_replay_identical':True},'module_calls':['whole_model'],'operator_calls':[]}
                audit_store.commit(index,'evaluation',verified)
            assert verified['metrics']=={'batch':record,'raw_logits_replay_identical':True}
            with np.load(audit_store.root/verified['output']['path'],allow_pickle=False) as a:
                if not synthetic:
                    if spec['model_type']=='gpt2':unit_sums[record['article']]+=math.fsum(a['token_NLL'])
                    else:unit_sums[record['review_start']:record['review_stop']]=1-a['correct']
            total_batches+=1
        assert len(result['evaluation'])==total_batches
        if not synthetic:assert int(unit_sizes.sum())==(283225 if spec['model_type']=='gpt2' else 25000)
        audit={'status':'complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed',
            'source_sha256':sha(__file__),'protocol_sha256':sha(args.protocol),'results_sha256':sha(root/'results.json'),
            'case':args.case,'case_spec':case,'synthetic_only':synthetic,'model_key':case['model_key'],
            'canonical_parameters':len(points),'unique_parameter_scalars':package['unique_learned_scalars'],
            'model_point_sha256':point_hash,'evaluation_batches':total_batches,
            'unit_metric_sums':unit_sums.tolist(),'unit_sizes':unit_sizes.tolist(),
            'metric':'synthetic_not_task_quality' if synthetic else ('NLL' if spec['model_type']=='gpt2' else 'classification_error'),
            'mean_metric':float(unit_sums.sum()/unit_sizes.sum()),
            'resources':{key:sum(r[key] for r in resources) for key in ['logical_pbits','retained_device_evaluations','warmup_device_evaluations','logical_retained_device_evaluations']},
            'elapsed_seconds_this_replay_attempt':time.monotonic()-begin,
            'scope':'All weights independently decoded from saved counts or analytic controls and every full outputbatch hashreplayed; no resampling, extraMCseed, R2or physicaldeviceadvantageclaim.'}
        store._atomic_json(audit_path,audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['unit_metric_sums','unit_sizes']}))
    except Exception as error:
        (attempt/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2)+'\n');raise

if __name__=='__main__':main()
