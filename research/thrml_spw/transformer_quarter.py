"""Development-only three-block native conversion, fixed finite controls.

Original development splits are reused explicitly; completed confirmation text
is never loaded. Calibration selects T, retaining every failed budget.
"""
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .transformer_block_gpu import sha,save_array,save_json,write_csv
from .sequential_transformer import SequentialRunner,projection_targets
from .block_quantization_controls import quantize
from .transformer_control import expected_weights
from .gpu_environment import cuda_environment


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    root=Path('results/thrml_spw');out=root/'transformer_quarter';out.mkdir(exist_ok=True)
    data_path=Path('data/thrml_spw/transformer_projection/export.npz');data=np.load(data_path)
    prepared=json.loads((root/'transformer_preparation/manifest.json').read_text());assert sha(data_path)==prepared['export_sha256']
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{prepared['config']['model_revision']}"
    targets=projection_targets([0,1,2]);model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();runner=SequentialRunner(model,targets)
    n=sum(w.size for w in runner.weights);assert n==21233664;count=len(targets);assert count==12
    protocol={'model':prepared['config']['model'],'revision':prepared['config']['model_revision'],'dataset_revision':prepared['dataset_revision'],
        'targets_in_forward_order':targets,'converted_matrix_weights':n,'model_parameters':prepared['total_parameters'],'converted_fraction':n/prepared['total_parameters'],
        'deterministic_converted_module_biases':sum(b.size for b in runner.biases),'matrix_shapes':[list(w.shape) for w in runner.weights],
        'input_data_path':str(data_path),'input_export_sha256':sha(data_path),'development_only':True,'completed_confirmation_data_used':False,
        'field_bits':8,'field_bound':4.,'input_block_size':128,'sampling_seeds':list(range(5)),
        'calibration_sample_search':[256,512,1024],'contexts_per_weight_bank_batch':8,'calibration_domain':970730,'test_domain':980730,
        'selection':'Five native validation seeds per ascending T; first mean CE increase<=log(1.05) and token-accuracy loss<=0.01. If none pass, retain largest as a failed diagnostic. Five original-test development seeds at selected T, no test-based selection. No completed confirmation windows loaded.',
        'scope':'All twelve projection matrices in first three blocks (25.92% model), finite uniform dimensionless Ising-b controls and local128 affine ranges. Biases, LayerNorm, embeddings/head and last three blocks deterministic. Software THRML/CUDA external arithmetic, not hardware proof.',
        'driver_source_sha256':sha(__file__),'runner_source_sha256':sha(Path(__file__).with_name('sequential_transformer.py'))}
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Do not overwrite a changed research protocol'
    else:save_json(pp,protocol)
    master=out/'weights.npz';arrays={**{f'weight_{i}':w for i,w in enumerate(runner.weights)},**{f'bias_{i}':b for i,b in enumerate(runner.biases)}}
    if master.exists():
        prior=np.load(master)
        for k,v in arrays.items():np.testing.assert_array_equal(prior[k],v)
    else:np.savez_compressed(master,**arrays)
    baselines={};base_rows=[];base_raw={};started=time.monotonic()
    for split in ('validation','test'):
        for condition in ('FP32','affine4_block128','affine8_block128','programmed_mean8'):
            try:
                if condition!='FP32':
                    with torch.no_grad():
                        for module,w in zip(runner.modules,runner.weights):
                            q=expected_weights(w,8,4.) if condition=='programmed_mean8' else quantize(w,4 if condition=='affine4_block128' else 8,'affine_zero',128)
                            module.weight.copy_(torch.from_numpy(q))
                result=runner.evaluate(data[split+'_tokens']);base_rows.append({'split':split,'condition':condition,**result['scores']})
                base_raw[f'{split}_{condition}_nll']=result['nll'];base_raw[f'{split}_{condition}_predictions']=result['predictions']
                if condition=='FP32':
                    np.testing.assert_array_equal(result['nll'],data[split+'_reference_token_nll']);baselines[split]=result['scores']
                print(f'Quarter baseline {split} {condition}: {result["scores"]}',flush=True)
            finally:runner.restore()
    write_csv(out/'baseline_metrics.csv',base_rows);np.savez_compressed(out/'baseline_tokens.npz',**base_raw)
    rows=[];prefixes=[];profiles=[]
    log=(out/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.block_control_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}, see worker.log')
        response=json.loads(line)
        if response['status']=='error':raise RuntimeError(str(response))
        return response
    try:
        assert reply()['status']=='ready'
        def run(split,t,seed):
            directory=out/'runs'/f'{split}_T{t}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            tokens=data[split+'_tokens'];replacements={};raw={};stages=[];checks=0
            for stage in range(count):
                result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks']
                raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions']
                prefixes.append({'split':split,'samples':t,'sampling_seed':seed,'converted_prefix':stage,**result['scores']})
                xp,rp,yp=[directory/f'stage{stage}_{name}.npy' for name in ('inputs','conditional_reference','native_output')]
                save_array(xp,result['inputs']);save_array(rp,result['outputs'])
                request={'weights':str(master),'inputs':str(xp),'reference':str(rp),'output':str(yp),'stage':stage,'samples':t,'seed':seed,
                    'domain':protocol['calibration_domain'] if split=='validation' else protocol['test_domain'],'contexts_per_batch':8,'field_bits':8,'field_bound':4.}
                worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok'
                meta=json.loads(Path(response['metadata']).read_text());assert meta['output_sha256']==sha(yp)
                replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'))
                stages.append({'stage':stage,'inputs':str(xp),'reference':str(rp),'output':str(yp),'metadata':response['metadata'],
                    'prior_native_output_sha256':[sha(directory/f'stage{k}_native_output.npy') for k in range(stage)]})
                print(f'Quarter native {split} T{t} seed{seed} stage{stage}/{count-1} elapsed={time.monotonic()-started:.1f}s',flush=True)
            final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==len(tokens)*count*(count+1)//2
            raw[f'prefix{count}_nll']=final['nll'];raw[f'prefix{count}_predictions']=final['predictions'];np.savez_compressed(directory/'prefix_tokens.npz',**raw)
            prefixes.append({'split':split,'samples':t,'sampling_seed':seed,'converted_prefix':count,**final['scores']})
            record={'status':'complete','split':split,'samples':t,'sampling_seed':seed,'scores':final['scores'],'stages':stages,'exact_prefix_input_checks':checks,
                'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
            save_json(directory/'run.json',record);rows.append({k:record[k] for k in ('split','samples','sampling_seed','free_node_updates','exact_prefix_input_checks')}|final['scores'])
            write_csv(out/'native_metrics.csv',rows);write_csv(out/'prefix_metrics.csv',prefixes)
            print(f'Quarter complete {split} T{t} seed{seed}: {final["scores"]}',flush=True)
        chosen=None
        for t in protocol['calibration_sample_search']:
            for seed in range(5):run('validation',t,seed)
            group=[r for r in rows if r['split']=='validation' and r['samples']==t];ce=float(np.mean([r['cross_entropy'] for r in group]));acc=float(np.mean([r['token_accuracy'] for r in group]))
            passed=bool(ce-baselines['validation']['cross_entropy']<=np.log(1.05) and baselines['validation']['token_accuracy']-acc<=.01)
            profile={'samples':t,'cross_entropy_mean':ce,'relative_perplexity':float(np.exp(ce-baselines['validation']['cross_entropy'])),'token_accuracy_mean':acc,'passes':passed}
            profiles.append(profile);save_json(out/'calibration_progress.json',{'profiles':profiles,'test_used_in_selection':False});print(profile,flush=True)
            if passed:chosen=t;break
        selected=chosen if chosen is not None else profiles[-1]['samples'];selection={'selected_samples':selected,'gate_passed':chosen is not None,'profiles':profiles,'test_used_in_selection':False}
        save_json(out/'selection.json',selection)
        for seed in range(5):run('test',selected,seed)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    summaries=[]
    for split,t in sorted({(r['split'],r['samples']) for r in rows}):
        group=[r for r in rows if r['split']==split and r['samples']==t];ce=np.array([r['cross_entropy'] for r in group]);mean=float(ce.mean());margin=float(1.96*ce.std(ddof=1)/np.sqrt(5));delta=mean-baselines[split]['cross_entropy']
        summaries.append({'split':split,'samples':t,'cross_entropy_mean':mean,'cross_entropy_ci95_margin':margin,'perplexity':float(np.exp(mean)),'relative_perplexity':float(np.exp(delta)),
            'relative_perplexity_lower':float(np.exp(delta-margin)),'relative_perplexity_upper':float(np.exp(delta+margin)),
            'token_accuracy_mean':float(np.mean([r['token_accuracy'] for r in group])),'free_node_updates':group[0]['free_node_updates'],'sampling_rounds_per_bank_batch':count*t})
    write_csv(out/'summary.csv',summaries);save_json(out/'manifest.json',{'config':protocol,'weights_sha256':sha(master),'run_count':len(rows),'summary':summaries,'selection':selection,'elapsed_seconds':time.monotonic()-started,'all_student_inputs_regenerated_and_checked':True})
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
