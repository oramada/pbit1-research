"""Sequential quarter-model quality under native measured device calibration."""
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .sequential_transformer import SequentialRunner
from .transformer_block_gpu import sha,save_array,save_json,write_csv
from .block_quantization_controls import quantize
from .calibrated_transformer_controls import decode
from .gpu_environment import cuda_environment
from .conservative_selection import summarize_profile

OUT=Path('results/thrml_spw/calibrated_transformer')


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    design=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preparation_audit.json').read_text())['passed']
    assert sha(OUT/'tokens.npz')==design['tokens_sha256'];tokens=np.load(OUT/'tokens.npz')['tokens'];assert tokens.shape==(64,65)
    source=['calibrated_transformer.py','calibrated_transformer_worker.py','calibrated_transformer_native.py','calibrated_transformer_controls.py','sequential_transformer.py','transformer_block_gpu.py','conservative_selection.py']
    execution={'prepared_design_sha256':sha(OUT/'protocol.json'),'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in source},'scope':'Prepared no-model-scores field describes preparation only. This execution records the complete15predeclaredquarterforwardruns.'}
    ep=OUT/'execution_protocol.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    archive=Path('results/thrml_spw/source_snapshots');archive.mkdir(exist_ok=True)
    for n,h in execution['source_sha256'].items():
        p=archive/(h+'.py');data=Path(n).read_bytes()
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{design['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),design['targets_in_forward_order'])
    started=time.monotonic();base=[];br={}
    for condition in ['FP32','affine4_block128','affine8_block128','allfree_programmed_mean8']:
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for stage,(module,w) in enumerate(zip(runner.modules,runner.weights)):
                        if condition=='allfree_programmed_mean8':
                            a=np.load(design['control_artifacts'][stage]['path']);q=1/(1+np.exp(-2*decode(a['codes']).astype(float)))
                            qq=np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q;qq=qq[:w.shape[0]].astype(np.float32)
                        else:qq=quantize(w,4 if condition=='affine4_block128' else 8,'affine_zero',128)
                        module.weight.copy_(torch.from_numpy(qq))
            result=runner.evaluate(tokens);base.append({'condition':condition,**result['scores']});br[f'{condition}_nll']=result['nll'];br[f'{condition}_predictions']=result['predictions']
            print(f'Calibrated quarter baseline {condition}: {result["scores"]}',flush=True)
        finally:runner.restore()
    write_csv(OUT/'baseline_metrics.csv',base);np.savez_compressed(OUT/'baseline_tokens.npz',**br)
    log=(OUT/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.calibrated_transformer_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited{worker.poll()}; see worker.log')
        value=json.loads(line)
        if value['status']=='error':raise RuntimeError(str(value))
        return value
    rows=[];prefixes=[]
    try:
        assert reply()['status']=='ready'
        for seed in range(5):
            for condition in design['conditions']:
                folder=OUT/'runs'/f'{condition}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
                if rp.exists():
                    record=json.loads(rp.read_text());assert record['status']=='complete' and record['design_sha256']==sha(OUT/'protocol.json')
                    for stage in record['stages']:
                        assert sha(stage['output'])==json.loads(Path(stage['metadata']).read_text())['output_sha256']
                    rows.append({k:record[k] for k in ['split','condition','sampling_seed','free_node_updates','exact_prefix_input_checks']}|record['scores']);continue
                replacements={};raw={};stages=[];checks=0
                for stage in range(12):
                    result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks'];raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions']
                    prefixes.append({'condition':condition,'sampling_seed':seed,'converted_prefix':stage,**result['scores']})
                    xp,refp,yp=[folder/f'stage{stage}_{n}.npy' for n in ['inputs','conditional_reference','native_output']]
                    save_array(xp,result['inputs']);save_array(refp,result['outputs'])
                    r={'controls':design['control_artifacts'][stage]['path'],'calibration_folder':str(OUT/'calibration'),'inputs':str(xp),'reference':str(refp),'output':str(yp),'stage':stage,'seed':seed,'condition':condition,'samples':1024,'contexts_per_bank':8}
                    worker.stdin.write(json.dumps(r)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yp)==meta['output_sha256']
                    replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'));stages.append({'stage':stage,'inputs':str(xp),'reference':str(refp),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]})
                    save_json(folder/'progress.json',{'stage_completed':stage,'condition':condition,'seed':seed,'elapsed_seconds':time.monotonic()-started})
                    print(f'Calibrated quarter {condition} seed{seed} stage{stage}/11 elapsed={time.monotonic()-started:.1f}s',flush=True)
                final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==64*78
                raw['prefix12_nll']=final['nll'];raw['prefix12_predictions']=final['predictions'];np.savez_compressed(folder/'prefix_tokens.npz',**raw)
                record={'status':'complete','split':'test','condition':condition,'sampling_seed':seed,'scores':final['scores'],'stages':stages,'design_sha256':sha(OUT/'protocol.json'),'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
                save_json(rp,record);rows.append({k:record[k] for k in ['split','condition','sampling_seed','free_node_updates','exact_prefix_input_checks']}|final['scores']);prefixes.append({'condition':condition,'sampling_seed':seed,'converted_prefix':12,**final['scores']})
                write_csv(OUT/'native_metrics.csv',rows);write_csv(OUT/'prefix_metrics.csv',prefixes)
                print(f'Complete calibrated quarter {condition} seed{seed}: {final["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    assert len(rows)==15;profiles=[]
    for ci,condition in enumerate(design['conditions']):
        raw=[np.load(OUT/'runs'/f'{condition}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
        profile,draws=summarize_profile(np.stack([a['prefix12_nll'] for a in raw]),np.stack([a['prefix12_predictions'] for a in raw]),br['FP32_nll'],br['FP32_predictions'],tokens,1024,2026114701)
        np.savez_compressed(OUT/f'{condition}_uncertainty_draws.npz',**draws);profiles.append({'condition':condition,**profile})
    write_csv(OUT/'summary.csv',profiles)
    cal=list((OUT/'calibration').glob('*.json'));assert len(cal)==60
    result={'config':design,'execution':execution,'native_forward_runs':15,'native_operator_runs':180,'native_probe_windows':120,'profiles':profiles,
        'total_native_inference_free_evaluations':sum(r['free_node_updates'] for r in rows),'total_native_calibration_free_evaluations':sum(json.loads(p.read_text())['free_evaluations'] for p in cal),'elapsed_seconds':time.monotonic()-started,'scope':design['scope']}
    save_json(OUT/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['config','execution']},indent=2))


if __name__=='__main__':main()
