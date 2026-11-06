"""Native validation-only isolated-layer sensitivity; never scores test text."""
import json,subprocess,time
from pathlib import Path
import numpy as np
from .sample_allocation_model import runner,set_means
from .transformer_block_gpu import sha,save_json,save_array,write_csv
from .gpu_environment import cuda_environment
OUT=Path('results/thrml_spw/sample_allocation')

def main():
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preparation_audit.json').read_text())['passed'];assert sha(OUT/'tokens.npz')==p['tokens_sha256'];sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['sample_allocation_validation.py','sample_allocation_worker.py','sample_allocation_model.py','sequential_transformer.py','transformer_block_gpu.py','calibrated_transformer_native.py']};execution={'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sources};ep=OUT/'validation_execution.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    if (OUT/'validation_manifest.json').exists():
        m=json.loads((OUT/'validation_manifest.json').read_text());assert m['execution']==execution
        for c in m['cases']:assert sha(c['output'])==c['output_sha256'] and sha(c['tokens'])==c['tokens_sha256']
        print(json.dumps({'cached_complete':True,'new_native_samples':False}));return
    for name,h in sources.items():
        target=Path('results/thrml_spw/source_snapshots')/(h+'.py')
        if not target.exists():target.write_bytes(Path(name).read_bytes())
    tokens=np.load(OUT/'tokens.npz')['validation_tokens'];r=runner(p);fp=r.evaluate(tokens);set_means(r,p);ref=r.evaluate(tokens);np.savez_compressed(OUT/'validation_baseline.npz',FP32_nll=fp['nll'],FP32_predictions=fp['predictions'],mean_nll=ref['nll'],mean_predictions=ref['predictions']);save_json(OUT/'validation_baseline.json',{'FP32':fp['scores'],'programmed_mean8_all24':ref['scores']});work=OUT/'validation/work';work.mkdir(parents=True,exist_ok=True);xp=work/'current_inputs.npy';log=(OUT/'validation_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.sample_allocation_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment());start=time.monotonic();cases=[];rows=[]
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}')
        result=json.loads(line)
        if result['status']=='error':raise RuntimeError(str(result))
        return result
    try:
        assert reply()['status']=='ready'
        for stage in range(24):
            captured=r.evaluate(tokens,capture=stage);np.testing.assert_array_equal(captured['nll'],ref['nll']);x=captured['inputs'];save_array(xp,x)
            for seed in range(5):
                folder=OUT/'validation/runs'/f'stage{stage}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
                if rp.exists():
                    case=json.loads(rp.read_text());assert case['status']=='complete' and case['protocol_sha256']==sha(OUT/'protocol.json') and sha(case['output'])==case['output_sha256'] and sha(case['tokens'])==case['tokens_sha256']
                else:
                    yp=folder/'native_output.npy';request={'phase':'validation','condition':'isolated','samples':256,'stage':stage,'seed':seed,'inputs':str(xp),'input_sha256':sha(xp),'output':str(yp)};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yp)==meta['output_sha256'];value=r.evaluate(tokens,{stage:(x,np.load(yp,mmap_mode='r'))});assert value['input_checks']==32;raw=folder/'token_metrics.npz';np.savez_compressed(raw,nll=value['nll'],predictions=value['predictions']);case={'status':'complete','stage':stage,'seed':seed,'samples':256,'inputs_sha256':sha(xp),'output':str(yp),'output_sha256':sha(yp),'metadata':response['metadata'],'tokens':str(raw),'tokens_sha256':sha(raw),'scores':value['scores'],'protocol_sha256':sha(OUT/'protocol.json'),'observed_node_updates':meta['free_node_updates']};save_json(rp,case)
                cases.append(case);rows.append({'stage':stage,'seed':seed,**case['scores'],'observed_node_updates':case['observed_node_updates']});print(f'Validation-only stage{stage} seed{seed}, CE={case["scores"]["cross_entropy"]:.6f}, elapsed={time.monotonic()-start:.1f}s',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    xp.unlink(missing_ok=True);assert len(cases)==120;write_csv(OUT/'validation_metrics.csv',rows);m={'execution':execution,'cases':cases,'native_operators':120,'native_observed_updates':sum(c['observed_node_updates'] for c in cases),'baseline_archive_sha256':sha(OUT/'validation_baseline.npz'),'elapsed_seconds':time.monotonic()-start,'test_scores_computed':False,'scope':'Only32 new validation windows, five nativeMC seeds per isolated layer; all other converted matrices analytic programmedmean. No test model scoring or allocation decisions yet.'};save_json(OUT/'validation_manifest.json',m);print(json.dumps({k:v for k,v in m.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
