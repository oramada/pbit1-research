"""Prospectively specified larger-calibration, conservative native quarter study."""
import json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .transformer_block_gpu import sha,save_array,save_json,write_csv
from .sequential_transformer import SequentialRunner
from .block_quantization_controls import quantize
from .transformer_control import expected_weights
from .gpu_environment import cuda_environment
from .conservative_selection import summarize_profile


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    root=Path('results/thrml_spw');prep=root/'quarter_conservative_preparation';out=root/'quarter_conservative';out.mkdir(exist_ok=True)
    design_path=prep/'protocol.json';design=json.loads(design_path.read_text());assert json.loads((prep/'reservation_audit.json').read_text())['passed']
    data_path=Path('data/thrml_spw/quarter_conservative/tokens.npz');assert sha(data_path)==design['tokens_sha256'];data=np.load(data_path)
    assert sha(root/'quarter_confirmation/manifest.json')==design['motivating_failed_confirmation_sha256']
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{design['model_revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),design['targets_in_forward_order'])
    master=out/'weights.npz'
    if not master.exists():shutil.copyfile(root/'transformer_quarter/weights.npz',master)
    assert sha(master)==sha(root/'transformer_quarter/weights.npz')
    protocol={**design,'revision':design['model_revision'],'input_data_path':str(data_path),'input_export_sha256':sha(data_path),'weights_sha256':sha(master),
        'predeclared_design_sha256':sha(design_path),'reservation_audit_sha256':sha(prep/'reservation_audit.json'),
        'driver_source_sha256':sha(__file__),'runner_source_sha256':sha(Path(__file__).with_name('sequential_transformer.py')),
        'selection_source_sha256':sha(Path(__file__).with_name('conservative_selection.py')),
        'execution_scope':'Actual native quarter-model execution. no_model_scores_computed in the inherited prospective design describes preparation only; run manifests below record execution.'}
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Existing protocol cannot be changed'
    else:save_json(pp,protocol)
    started=time.monotonic();baseline_rows=[];base_raw={};rows=[];prefixes=[];profiles=[];baselines={}
    def baseline(split):
        tokens=data[split+'_tokens'];assert tokens.shape==(128,65)
        for condition in ['FP32','affine4_block128','affine8_block128','programmed_mean8']:
            try:
                if condition!='FP32':
                    with torch.no_grad():
                        for module,w in zip(runner.modules,runner.weights):
                            q=expected_weights(w,8,4.) if condition=='programmed_mean8' else quantize(w,4 if condition=='affine4_block128' else 8,'affine_zero',128)
                            module.weight.copy_(torch.from_numpy(q))
                result=runner.evaluate(tokens);baseline_rows.append({'split':split,'condition':condition,**result['scores']})
                base_raw[f'{split}_{condition}_nll']=result['nll'];base_raw[f'{split}_{condition}_predictions']=result['predictions']
                if condition=='FP32':baselines[split]=(result['nll'],result['predictions'])
                print(f'Conservative baseline {split} {condition}: {result["scores"]}',flush=True)
            finally:runner.restore()
        write_csv(out/'baseline_metrics.csv',baseline_rows);np.savez_compressed(out/'baseline_tokens.npz',**base_raw)
    baseline('validation')
    log=(out/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.block_cached_control_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}, see worker.log')
        value=json.loads(line)
        if value['status']=='error':raise RuntimeError(str(value))
        return value
    try:
        assert reply()['status']=='ready'
        def run(split,t,seed):
            folder=out/'runs'/f'{split}_T{t}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);tokens=data[split+'_tokens'];replacements={};raw={};stages=[];checks=0
            for stage in range(12):
                result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks']
                raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions'];prefixes.append({'split':split,'samples':t,'sampling_seed':seed,'converted_prefix':stage,**result['scores']})
                xp,rp,yp=[folder/f'stage{stage}_{name}.npy' for name in ['inputs','conditional_reference','native_output']]
                save_array(xp,result['inputs']);save_array(rp,result['outputs'])
                request={'weights':str(master),'inputs':str(xp),'reference':str(rp),'output':str(yp),'stage':stage,'samples':t,'seed':seed,
                    'domain':design['validation_domain'] if split=='validation' else design['test_domain'],'contexts_per_batch':8,'field_bits':8,'field_bound':4.}
                worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok'
                meta=json.loads(Path(response['metadata']).read_text());assert meta['output_sha256']==sha(yp)
                replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'));stages.append({'stage':stage,'inputs':str(xp),'reference':str(rp),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]})
                print(f'Conservative native {split} T{t} seed{seed} stage{stage}/11 elapsed={time.monotonic()-started:.1f}s',flush=True)
            final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==128*78
            raw['prefix12_nll']=final['nll'];raw['prefix12_predictions']=final['predictions'];np.savez_compressed(folder/'prefix_tokens.npz',**raw);prefixes.append({'split':split,'samples':t,'sampling_seed':seed,'converted_prefix':12,**final['scores']})
            record={'status':'complete','split':split,'samples':t,'sampling_seed':seed,'scores':final['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
            save_json(folder/'run.json',record);rows.append({k:record[k] for k in ['split','samples','sampling_seed','free_node_updates','exact_prefix_input_checks']}|final['scores']);write_csv(out/'native_metrics.csv',rows);write_csv(out/'prefix_metrics.csv',prefixes)
            print(f'Conservative complete {split} T{t} seed{seed}: {final["scores"]}',flush=True)
        def summarize(split,t,bootstrap_seed):
            raw=[np.load(out/'runs'/f'{split}_T{t}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
            record,draws=summarize_profile(np.stack([r['prefix12_nll'] for r in raw]),np.stack([r['prefix12_predictions'] for r in raw]),*baselines[split],data[split+'_tokens'],t,bootstrap_seed)
            np.savez_compressed(out/f'{split}_T{t}_selection_draws.npz',**draws)
            record['split']=split;record['free_node_updates']=next(r['free_node_updates'] for r in rows if r['split']==split and r['samples']==t)
            return record
        selected=None
        for t in design['sample_search']:
            for seed in design['sampling_seeds']:run('validation',t,seed)
            profile=summarize('validation',t,design['selection_bootstrap']['seed']);profiles.append(profile)
            save_json(out/'calibration_progress.json',{'profiles':profiles,'test_scores_computed':False});print('Conservative calibration',profile,flush=True)
            if profile['passes']:selected=t;break
        chosen=selected if selected is not None else design['sample_search'][-1]
        selection={'selected_samples':chosen,'gate_passed':selected is not None,'profiles':profiles,'test_scores_used_in_selection':False,'predeclared_design_sha256':protocol['predeclared_design_sha256']};save_json(out/'selection.json',selection)
        # Only now expose confirmation scores; selection has been written atomically.
        baseline('test')
        for seed in design['sampling_seeds']:run('test',chosen,seed)
        test=summarize('test',chosen,20261112)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    write_csv(out/'summary.csv',profiles+[test]);save_json(out/'manifest.json',{'config':protocol,'selection':selection,'profiles':profiles,'confirmation':test,'run_count':len(rows),
        'no_test_selection':True,'all_student_inputs_regenerated_and_checked':True,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'selection':selection,'confirmation':test},indent=2),flush=True)

if __name__=='__main__':main()
