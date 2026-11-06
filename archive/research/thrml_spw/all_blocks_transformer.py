"""Fixed-regime all-six-block native Transformer scaling study."""
import json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .sequential_transformer import SequentialRunner
from .transformer_block_gpu import sha,save_json,save_array,write_csv
from .block_quantization_controls import quantize
from .calibrated_transformer_controls import decode
from .gpu_environment import cuda_environment
from .all_blocks_statistics import main as statistics
OUT=Path('results/thrml_spw/all_blocks_clean')

def main():
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preparation_audit.json').read_text())['passed'];assert sha(OUT/'tokens.npz')==p['tokens_sha256']
    for name,h in p['source_sha256'].items():assert sha(name)==h
    names=['all_blocks_transformer.py','all_blocks_worker.py','all_blocks_statistics.py','sequential_transformer.py','transformer_block_gpu.py','calibrated_transformer_native.py'];sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in names};execution={'design_sha256':sha(OUT/'protocol.json'),'source_sha256':sources}
    ep=OUT/'execution_protocol.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    if (OUT/'manifest.json').exists():
        m=json.loads((OUT/'manifest.json').read_text());assert m['config']==p and m['execution']==execution
        for seed in range(5):
            folder=OUT/'runs'/f'seed{seed}';r=json.loads((folder/'run.json').read_text());assert sha(folder/'prefix_tokens.npz')==r['token_archive_sha256']
            for s in r['stages']:assert sha(s['output'])==json.loads(Path(s['metadata']).read_text())['output_sha256']
        print(json.dumps({'cached_complete':True,'new_native_sampling':False,'manifest_sha256':sha(OUT/'manifest.json')}));return
    archive=Path('results/thrml_spw/source_snapshots')
    for name,h in sources.items():
        target=archive/(h+'.py')
        if target.exists():assert target.read_bytes()==Path(name).read_bytes()
        else:target.write_bytes(Path(name).read_bytes())
    assert shutil.disk_usage(OUT).free>1800000000,'Insufficient space for native evidence and temporary audit inputs'
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);tokens=np.load(OUT/'tokens.npz')['tokens'];snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}";runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order']);start=time.monotonic();base=[];rawbase={}
    for condition in ['FP32','affine4_all24','affine8_all24','programmed_mean8_first12','programmed_mean8_all24']:
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for stage,(module,w) in enumerate(zip(runner.modules,runner.weights)):
                        if condition=='programmed_mean8_first12' and stage>=12:continue
                        if condition.startswith('programmed'):
                            a=np.load(p['control_artifacts'][stage]['path']);q=1/(1+np.exp(-2*decode(a['codes']).astype(float)));ww=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q).astype(np.float32)
                        else:ww=quantize(w,4 if condition=='affine4_all24' else 8,'affine_zero',128)
                        module.weight.copy_(torch.from_numpy(ww))
            r=runner.evaluate(tokens);base.append({'condition':condition,**r['scores']});rawbase[condition+'_nll']=r['nll'];rawbase[condition+'_predictions']=r['predictions'];print(f'Baseline {condition}: {r["scores"]}',flush=True)
        finally:runner.restore()
    write_csv(OUT/'baseline_metrics.csv',base);np.savez_compressed(OUT/'baseline_tokens.npz',**rawbase)
    work=OUT/'work';work.mkdir(exist_ok=True);xp=work/'current_inputs.npy';log=(OUT/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.all_blocks_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    try:
        assert reply()['status']=='ready'
        for seed in range(5):
            folder=OUT/'runs'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
            if rp.exists():
                r=json.loads(rp.read_text());assert r['status']=='complete' and r['protocol_sha256']==sha(OUT/'protocol.json') and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'];continue
            replacements={};raw={};stages=[];checks=0
            for stage in range(24):
                result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks'];raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions']
                if stage==0:np.testing.assert_array_equal(result['nll'],rawbase['FP32_nll'])
                save_array(xp,result['inputs']);yp=folder/f'stage{stage}_native_output.npy';request={'stage':stage,'seed':seed,'samples':1024,'inputs':str(xp),'input_sha256':sha(xp),'output':str(yp)};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yp)==meta['output_sha256'] and meta['signature']['inputs_sha256']==sha(xp)
                replacements[stage]=(result['inputs'],np.load(yp,mmap_mode='r'));stages.append({'stage':stage,'inputs_sha256':sha(xp),'input_shape':list(result['inputs'].shape),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]});print(f'All-block native seed{seed} stage{stage}/23 elapsed={time.monotonic()-start:.1f}s',flush=True)
            final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==32*300;raw['prefix24_nll']=final['nll'];raw['prefix24_predictions']=final['predictions'];np.savez_compressed(folder/'prefix_tokens.npz',**raw)
            r={'status':'complete','sampling_seed':seed,'scores':final['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages),'token_archive_sha256':sha(folder/'prefix_tokens.npz'),'protocol_sha256':sha(OUT/'protocol.json')};save_json(rp,r);save_json(OUT/'execution_progress.json',{'last_complete_seed':seed,'elapsed_seconds':time.monotonic()-start});print(f'Completed all-block seed{seed}: {final["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    xp.unlink(missing_ok=True);rows=[];prefix=[]
    for seed in range(5):
        folder=OUT/'runs'/f'seed{seed}';r=json.loads((folder/'run.json').read_text());a=np.load(folder/'prefix_tokens.npz');rows.append({'sampling_seed':seed,**r['scores'],'free_node_updates':r['free_node_updates']})
        for k in range(25):
            ce=float(a[f'prefix{k}_nll'].astype(float).mean());prefix.append({'sampling_seed':seed,'converted_prefix':k,'cross_entropy':ce,'perplexity':float(np.exp(ce)),'token_accuracy':float(np.mean(a[f'prefix{k}_predictions']==tokens[:,1:]))})
    write_csv(OUT/'native_metrics.csv',rows);write_csv(OUT/'prefix_metrics.csv',prefix);summary,pair=statistics();m={'config':p,'execution':execution,'native_full_forward_runs':5,'native_operator_runs':120,'native_probe_windows':0,'first12_prefix_shared_without_extra_sampling':True,'native_free_node_updates':sum(r['free_node_updates'] for r in rows),'summary':summary,'paired_contrast':pair,'elapsed_seconds':time.monotonic()-start};save_json(OUT/'manifest.json',m);print(json.dumps({k:v for k,v in m.items() if k not in ['config','execution']},indent=2))
if __name__=='__main__':main()
