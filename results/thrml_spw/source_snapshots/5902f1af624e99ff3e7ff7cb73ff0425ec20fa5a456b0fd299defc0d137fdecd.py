"""Fresh full-model comparison after audited validation-only allocation is frozen."""
import json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
from .sample_allocation_model import runner,set_means
from .transformer_block_gpu import sha,save_json,save_array,write_csv
from .block_quantization_controls import quantize
from .gpu_environment import cuda_environment
from .sample_allocation_statistics import main as statistics
from .packed_float_outputs import load_output
OUT=Path('results/thrml_spw/sample_allocation')

def main():
    p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'allocation.json').read_text());selection=json.loads((OUT/'selection_audit.json').read_text());assert selection['passed'] and sha(OUT/'allocation.json')==selection['allocation_sha256'];assert json.loads((OUT/'validation_evidence_audit.json').read_text())['passed'];assert sha(OUT/'tokens.npz')==p['tokens_sha256']
    for name,h in {**p['source_sha256'],**a['source_sha256']}.items():assert sha(name)==h
    names=['sample_allocation_test.py','sample_allocation_test_worker.py','packed_float_outputs.py','sample_allocation_statistics.py','sample_allocation_model.py','sequential_transformer.py','calibrated_transformer_native.py'];sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in names};execution={'protocol_sha256':sha(OUT/'protocol.json'),'allocation_sha256':sha(OUT/'allocation.json'),'selection_audit_sha256':sha(OUT/'selection_audit.json'),'source_sha256':sources};ep=OUT/'test_execution.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    if (OUT/'test_manifest.json').exists():
        m=json.loads((OUT/'test_manifest.json').read_text());assert m['execution']==execution
        for condition in ['uniform','allocated']:
            for seed in range(5):
                folder=OUT/'test/runs'/f'{condition}_seed{seed}';r=json.loads((folder/'run.json').read_text());assert sha(folder/'prefix_tokens.npz')==r['token_archive_sha256']
                for s in r['stages']:assert sha(s['output'])==json.loads(Path(s['metadata']).read_text())['output_sha256']
        print(json.dumps({'cached_complete':True,'new_native_sampling':False,'manifest_sha256':sha(OUT/'test_manifest.json')}));return
    for name,h in sources.items():
        target=Path('results/thrml_spw/source_snapshots')/(h+'.py')
        if not target.exists():target.write_bytes(Path(name).read_bytes())
    assert shutil.disk_usage(OUT).free>3100000000
    tokens=np.load(OUT/'tokens.npz')['test_tokens'];r=runner(p);base=[];rawbase={};start=time.monotonic()
    for condition in ['FP32','affine4_all24','affine8_all24','programmed_mean8_all24']:
        try:
            if condition=='programmed_mean8_all24':set_means(r,p)
            elif condition!='FP32':
                with torch.no_grad():
                    for module,w in zip(r.modules,r.weights):module.weight.copy_(torch.from_numpy(quantize(w,4 if condition=='affine4_all24' else 8,'affine_zero',128)))
            value=r.evaluate(tokens);base.append({'condition':condition,**value['scores']});rawbase[condition+'_nll']=value['nll'];rawbase[condition+'_predictions']=value['predictions'];print(f'Fresh test baseline {condition}: {value["scores"]}',flush=True)
        finally:r.restore()
    write_csv(OUT/'test_baseline_metrics.csv',base);np.savez_compressed(OUT/'test_baseline_tokens.npz',**rawbase);work=OUT/'test/work';work.mkdir(parents=True,exist_ok=True);xp=work/'current_inputs.npy';log=(OUT/'test_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.sample_allocation_test_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}')
        result=json.loads(line)
        if result['status']=='error':raise RuntimeError(str(result))
        return result
    try:
        assert reply()['status']=='ready'
        for seed in range(5):
            for condition in ['uniform','allocated']:
                folder=OUT/'test/runs'/f'{condition}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
                if rp.exists():
                    record=json.loads(rp.read_text());assert record['status']=='complete' and record['execution']==execution and sha(folder/'prefix_tokens.npz')==record['token_archive_sha256'];continue
                replacements={};raw={};stages=[];checks=0
                for stage in range(24):
                    value=r.evaluate(tokens,replacements,capture=stage);checks+=value['input_checks'];raw[f'prefix{stage}_nll']=value['nll'];raw[f'prefix{stage}_predictions']=value['predictions']
                    if stage==0:np.testing.assert_array_equal(value['nll'],rawbase['FP32_nll'])
                    save_array(xp,value['inputs']);t=1024 if condition=='uniform' else a['samples'][stage];yp=folder/f'stage{stage}_native_output.npz';request={'phase':'test','condition':condition,'stage':stage,'seed':seed,'samples':t,'inputs':str(xp),'input_sha256':sha(xp),'output':str(yp)};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yp)==meta['output_sha256'];replacements[stage]=(value['inputs'],load_output(yp));stages.append({'stage':stage,'samples':t,'inputs_sha256':sha(xp),'input_shape':list(value['inputs'].shape),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npz') for k in range(stage)]});print(f'Test {condition} seed{seed} stage{stage} T{t} elapsed={time.monotonic()-start:.1f}s',flush=True)
                value=r.evaluate(tokens,replacements);checks+=value['input_checks'];assert checks==9600;raw['prefix24_nll']=value['nll'];raw['prefix24_predictions']=value['predictions'];np.savez_compressed(folder/'prefix_tokens.npz',**raw);updates=sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages);assert updates==4*42467328*1024;record={'status':'complete','condition':condition,'seed':seed,'scores':value['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':updates,'token_archive_sha256':sha(folder/'prefix_tokens.npz'),'execution':execution};save_json(rp,record);save_json(OUT/'test_progress.json',{'last_complete_condition':condition,'last_complete_seed':seed,'elapsed_seconds':time.monotonic()-start});print(f'Completed test {condition} seed{seed}: {value["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    xp.unlink(missing_ok=True);rows=[];prefix=[]
    for condition in ['uniform','allocated']:
        for seed in range(5):
            folder=OUT/'test/runs'/f'{condition}_seed{seed}';record=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');rows.append({'condition':condition,'seed':seed,**record['scores'],'free_node_updates':record['free_node_updates']})
            for k in range(25):
                ce=float(raw[f'prefix{k}_nll'].astype(float).mean());prefix.append({'condition':condition,'seed':seed,'converted_prefix':k,'cross_entropy':ce,'perplexity':float(np.exp(ce)),'token_accuracy':float(np.mean(raw[f'prefix{k}_predictions']==tokens[:,1:]))})
    write_csv(OUT/'test_native_metrics.csv',rows);write_csv(OUT/'test_prefix_metrics.csv',prefix);summary,pair=statistics();save_json(OUT/'test_manifest.json',{'execution':execution,'native_model_cases':10,'native_operators':240,'native_observed_updates':sum(v['free_node_updates'] for v in rows),'summary':summary,'paired_contrast':pair,'elapsed_seconds':time.monotonic()-start,'test_scores_used_for_selection':False,'scope':p['scope']});print(json.dumps({'summary':summary,'paired':pair},indent=2))
if __name__=='__main__':main()
