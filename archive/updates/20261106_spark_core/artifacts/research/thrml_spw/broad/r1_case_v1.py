"""Checkpointed full fixed-task R1 inference, with every sampled count retained."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from .r1_io_v1 import CaseStore,sha,array_sha,program,reconstructed,source_guard
from .r1_model_v1 import load_model,install,control,batches,forward,observations


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--case',required=True)
    args=ap.parse_args();p=json.loads(args.protocol.read_text());source_guard(p)
    case=p['cases'][args.case];spec=p['models'][case['model_key']]
    synthetic=p['status']=='actual_model_synthetic_preflight_no_task_scoring'
    assert (spec['data']['kind']=='synthetic')==synthetic
    root=(Path(p['output'])/'cases'/args.case).resolve()
    contract={'protocol_sha256':sha(args.protocol),'case':args.case,'case_spec':case,'model_spec':spec}
    store=CaseStore(root,contract)
    if (root/'results.json').exists():return
    if list((root/'attempts').glob('*/failure.json')):raise RuntimeError('Preserved program failure is terminal')
    attempt=store.new_attempt();begin=time.monotonic();worker=None;log=(attempt/'worker.log').open('x')
    def progress(stage,**extra):
        tmp=root/'progress.tmp';tmp.write_text(json.dumps({'stage':stage,'elapsed_seconds':time.monotonic()-begin,**extra})+'\n');tmp.replace(root/'progress.json')
    def event(record):
        with (attempt/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    try:
        progress('loading_pinned_BF16_model')
        model,parameters,package=load_model(spec);points=[];native=case['mode']=='native'
        for record in package['canonical_parameters']:
            index=record['index'];name=record['name'];progress('installing_complete_parameter_point',index=index,name=name)
            assert sha(record['program'])==record['program_sha256']
            if native:
                closed=store.load(index,'parameter')
                if closed is None:
                    if worker is None:
                        worker=subprocess.Popen([sys.executable,'-m','research.thrml_spw.gpu_environment','--',
                            '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.broad.r1_worker_v1'],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
                        ready=json.loads(worker.stdout.readline());assert ready['ready'] and ready['backend']=='gpu'
                    counts=attempt/f'parameter{index:04d}_counts.npz';decoded=attempt/'runtime_decoded.npy'
                    request={'program':record['program'],'program_sha256':record['program_sha256'],
                        'bits':case['bits'],'samples':case['samples'],'seed':case['seed'],'index':index,
                        'counts_output':str(counts),'decoded_output':str(decoded)}
                    event({'event':'sampling_parameter_started','index':index,'request':request})
                    worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();line=worker.stdout.readline()
                    if not line:raise RuntimeError('OfficialTHRML count worker terminated')
                    receipt=json.loads(line)
                    assert receipt['ok'] and receipt['finite'] and not receipt['reference_weights_supplied']
                    assert receipt['counts_sha256']==sha(counts) and receipt['program_sha256']==record['program_sha256']
                    value=np.load(decoded,allow_pickle=False)
                    assert value.dtype==np.float32 and array_sha(value)==receipt['decoded_FP32_sha256']
                    digest=install(parameters,name,value)
                    closed={'output':{'path':str(counts.relative_to(store.root)),'sha256':sha(counts)},
                        'metrics':{'name':name,'BF16_sha256':digest},'module_calls':[name],'operator_calls':[receipt]}
                    store.commit(index,'parameter',closed)
                    event({'event':'sampling_parameter_committed','index':index,'receipt':receipt})
                else:
                    with np.load(store.root/closed['output']['path'],allow_pickle=False) as a:counts=a['counts'].copy()
                    value=reconstructed(program(record['program']),counts,case['samples'])
                    digest=install(parameters,name,value);assert digest==closed['metrics']['BF16_sha256']
                points.append({'index':index,'name':name,'BF16_sha256':closed['metrics']['BF16_sha256']})
            else:
                digest=control(parameters,record,case['arm'])
                points.append({'index':index,'name':name,'BF16_sha256':digest})
        point_hash=hashlib.sha256(json.dumps(points,sort_keys=True).encode()).hexdigest()
        evaluated=[]
        for record,ids,mask,labels in batches(spec):
            index=record['index'];progress('fixed_task_forward',batch=index)
            closed=store.load(index,'evaluation')
            if closed is None:
                logits=forward(model,ids,mask);raw_hash=array_sha(logits)
                arrays=observations(record,ids,labels,logits)
                path=attempt/f'evaluation{index:05d}.npz'
                with path.open('xb') as f:np.savez_compressed(f,**arrays)
                closed={'output':{'path':str(path.relative_to(store.root)),'sha256':sha(path)},
                    'metrics':{'batch':record,'full_logits_FP32_sha256':raw_hash,'logits_shape':list(logits.shape),
                        'model_point_sha256':point_hash,'input_ids_sha256':array_sha(ids),
                        'attention_mask_sha256':array_sha(mask) if mask is not None else None},
                    'module_calls':['whole_model'],'operator_calls':[]}
                store.commit(index,'evaluation',closed)
            assert closed['metrics']['batch']==record and closed['metrics']['model_point_sha256']==point_hash
            evaluated.append(closed)
        result={'status':'complete_R1_case_produced_pending_independent_neural_replay',
            'protocol_sha256':sha(args.protocol),'case':args.case,'case_spec':case,'model_key':case['model_key'],
            'package_sha256':spec['package_sha256'],'synthetic_only':synthetic,'parameter_points':points,
            'model_point_sha256':point_hash,'evaluation':evaluated,'elapsed_seconds_this_attempt':time.monotonic()-begin,
            'attempts':[str(x.relative_to(root)) for x in sorted((root/'attempts').iterdir())],
            'scope':'One complete reconstructedmodel draw reused across all fixedexamples. Counts/programs/weight hashes retained. Original BF16model allocation and simulator counters remain; no R2or physicalmemory/energyadvantage claim. Partial/replayedwork retained in attempt events, never additionalMCseeds.'}
        store._atomic_json(root/'results.json',result);progress('complete_pending_independent_neural_replay')
    except Exception as error:
        (attempt/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2)+'\n');raise
    finally:
        if worker is not None:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.stdin.close()
                except BrokenPipeError:pass
            try:worker.wait(timeout=60)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        log.close()

if __name__=='__main__':main()
