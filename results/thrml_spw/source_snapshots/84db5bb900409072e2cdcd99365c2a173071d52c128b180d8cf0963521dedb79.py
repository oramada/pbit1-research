"""Fresh matched-budget Transformer confirmation of frozen weight-only probes."""
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .sequential_transformer import SequentialRunner
from .transformer_block_gpu import sha,save_json,save_array,write_csv
from .block_quantization_controls import quantize
from .probe_placement_math import decode
from .probe_placement_statistics import summarize
from .gpu_environment import cuda_environment
OUT=Path('results/thrml_spw/probe_placement')


def verify_cache(p,execution):
    m=json.loads((OUT/'manifest.json').read_text());assert m['config']==p and m['execution']==execution
    for method in p['methods']:
        for seed in range(5):
            folder=OUT/'runs'/f'{method}_seed{seed}';r=json.loads((folder/'run.json').read_text());assert r['status']=='complete' and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256']
            for s in r['stages']:
                meta=json.loads(Path(s['metadata']).read_text());assert sha(s['output'])==meta['output_sha256']
    for cp in (OUT/'calibration').glob('*.json'):
        r=json.loads(cp.read_text());assert sha(cp.with_suffix('.npz'))==r['archive_sha256']
    print(json.dumps({'cached_complete':True,'new_native_sampling':False,'manifest_sha256':sha(OUT/'manifest.json')}),flush=True)


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preparation_audit.json').read_text())['passed']
    for n,h in p['source_sha256'].items():assert sha(n)==h
    assert sha(OUT/'tokens.npz')==p['tokens_sha256'] and sha(OUT/'placement_designs.json')==p['placement_designs_sha256']
    source=['probe_placement_transformer.py','probe_placement_worker.py','probe_placement_statistics.py','probe_placement_math.py','sequential_transformer.py','transformer_block_gpu.py']
    execution={'design_sha256':sha(OUT/'protocol.json'),'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in source},'scope':'All choices frozen before native counts or new task scoring; input files temporary, exact regeneration required by independent audit.'}
    ep=OUT/'execution_protocol.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    if (OUT/'manifest.json').exists():verify_cache(p,execution);return
    archive=Path('results/thrml_spw/source_snapshots');archive.mkdir(exist_ok=True)
    for n,h in execution['source_sha256'].items():
        target=archive/(h+'.py');data=Path(n).read_bytes()
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
    tokens=np.load(OUT/'tokens.npz')['tokens'];snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order']);start=time.monotonic();base=[];rawbase={}
    for condition in ['FP32','affine4_block128','affine8_block128','programmed_mean8']:
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for stage,(module,w) in enumerate(zip(runner.modules,runner.weights)):
                        if condition=='programmed_mean8':
                            a=np.load(p['control_artifacts'][stage]['path']);q=1/(1+np.exp(-2*decode(a['codes']).astype(float)));ww=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q).astype(np.float32)
                        else:ww=quantize(w,4 if condition=='affine4_block128' else 8,'affine_zero',128)
                        module.weight.copy_(torch.from_numpy(ww))
            r=runner.evaluate(tokens);base.append({'condition':condition,**r['scores']});rawbase[condition+'_nll']=r['nll'];rawbase[condition+'_predictions']=r['predictions'];print(f'Fresh baseline {condition}: {r["scores"]}',flush=True)
        finally:runner.restore()
    write_csv(OUT/'baseline_metrics.csv',base);np.savez_compressed(OUT/'baseline_tokens.npz',**rawbase)
    work=OUT/'work';work.mkdir(exist_ok=True);xp=work/'current_inputs.npy';log=(OUT/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.probe_placement_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited{worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    try:
        assert reply()['status']=='ready'
        for seed in range(5):
            for method in p['methods']:
                folder=OUT/'runs'/f'{method}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
                if rp.exists():
                    r=json.loads(rp.read_text());assert r['status']=='complete' and r['protocol_sha256']==sha(OUT/'protocol.json') and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'];continue
                replacements={};raw={};stages=[];checks=0
                for stage in range(12):
                    result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks'];raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions']
                    if stage==0:np.testing.assert_array_equal(result['nll'],rawbase['FP32_nll'])
                    save_array(xp,result['inputs']);yh=folder/f'stage{stage}_native_output.npy';request={'method':method,'stage':stage,'seed':seed,'samples':1024,'inputs':str(xp),'input_sha256':sha(xp),'output':str(yh)}
                    worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yh)==meta['output_sha256'] and meta['signature']['inputs_sha256']==sha(xp)
                    # Previous runtime input arrays stay in RAM for exact causal
                    # checks. On disk only their immutable SHA256 is retained.
                    replacements[stage]=(result['inputs'],np.load(yh,mmap_mode='r'));stages.append({'stage':stage,'inputs_sha256':meta['signature']['inputs_sha256'],'input_shape':list(result['inputs'].shape),'output':str(yh),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]})
                    print(f'Matched probes {method} seed{seed} stage{stage}/11 elapsed={time.monotonic()-start:.1f}s',flush=True)
                final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==64*78;raw['prefix12_nll']=final['nll'];raw['prefix12_predictions']=final['predictions'];np.savez_compressed(folder/'prefix_tokens.npz',**raw)
                r={'status':'complete','split':'test','method':method,'sampling_seed':seed,'scores':final['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages),'token_archive_sha256':sha(folder/'prefix_tokens.npz'),'protocol_sha256':sha(OUT/'protocol.json')};save_json(rp,r)
                save_json(OUT/'execution_progress.json',{'last_complete_method':method,'last_complete_seed':seed,'elapsed_seconds':time.monotonic()-start,'scope':'Native phase still in progress until all ten case manifests and final manifest exist.'})
                print(f'Complete matched probes {method} seed{seed}: {final["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    if xp.exists():xp.unlink() # Temporary, deterministically regenerable IPC copy.
    rows=[];prefix=[]
    for method in p['methods']:
        for seed in range(5):
            folder=OUT/'runs'/f'{method}_seed{seed}';r=json.loads((folder/'run.json').read_text());a=np.load(folder/'prefix_tokens.npz');rows.append({'method':method,'sampling_seed':seed,**r['scores'],'free_node_updates':r['free_node_updates']})
            for stage in range(13):
                ce=float(a[f'prefix{stage}_nll'].astype(float).mean());prefix.append({'method':method,'sampling_seed':seed,'converted_prefix':stage,'cross_entropy':ce,'perplexity':float(np.exp(ce)),'token_accuracy':float(np.mean(a[f'prefix{stage}_predictions']==tokens[:,1:]))})
    assert len(rows)==10;write_csv(OUT/'native_metrics.csv',rows);write_csv(OUT/'prefix_metrics.csv',prefix)
    profiles,primary=summarize(OUT);cal=list((OUT/'calibration').glob('*.json'));assert len(cal)==120
    result={'config':p,'execution':execution,'native_forward_runs':10,'native_operator_runs':120,'native_probe_windows':240,'profiles':profiles,'primary':primary,'native_inference_free_evaluations':sum(r['free_node_updates'] for r in rows),'native_calibration_free_evaluations':sum(json.loads(c.read_text())['free_node_evaluations'] for c in cal),'elapsed_seconds':time.monotonic()-start,'scope':p['scope']}
    save_json(OUT/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['config','execution']},indent=2))


if __name__=='__main__':main()
