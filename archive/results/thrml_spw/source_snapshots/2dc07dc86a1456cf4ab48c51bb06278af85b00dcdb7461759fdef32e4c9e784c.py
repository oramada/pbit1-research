"""Frozen confirmation of native first-block weight banks on reserved text.

No precision, grouping, operator or sampling search is permitted in this run.
"""
import json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .transformer_block_gpu import BlockRunner,sha,save_array,save_json,write_csv
from .block_quantization_controls import quantize
from .gpu_environment import cuda_environment


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    root=Path('results/thrml_spw');out=root/'transformer_confirmation';out.mkdir(exist_ok=True)
    reserve=root/'transformer_confirmation_preparation';reservation=json.loads((reserve/'protocol.json').read_text())
    reservation_audit=json.loads((reserve/'reservation_audit.json').read_text());assert reservation_audit['passed']
    token_path=Path('data/thrml_spw/transformer_confirmation/tokens.npz');assert sha(token_path)==reservation['tokens_sha256']
    data=np.load(token_path);tokens=data['tokens'];assert tokens.shape==(128,65)
    frozen=root/'transformer_control/selection.json';selection=json.loads(frozen.read_text());assert selection['gate_passed']
    bits=selection['selected_bits'];assert bits==8
    source=root/'transformer_block0_group128';base=json.loads((source/'protocol.json').read_text())
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{base['revision']}"
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();runner=BlockRunner(model)
    master=out/'weights.npz'
    if not master.exists():shutil.copyfile(source/'weights.npz',master)
    assert sha(master)==sha(source/'weights.npz')
    protocol={**base,'scope':'Frozen confirmation on previously unscored reserved tiles from the same WikiText-2 test corpus; first-block four projection matrices only, not full-model conversion or hardware proof.',
        'input_data_path':str(token_path),'input_tokens_key':'tokens','input_export_sha256':sha(token_path),
        'confirmation_windows':128,'target_tokens':8192,'conditions':['FP32','affine_zero_block128_4','native_float','native_finite8'],
        'samples':256,'sampling_seeds':list(range(5)),'contexts_per_weight_bank_batch':8,'confirmation_domain':960730,
        'selected_field_bits':bits,'field_bound':4.,'selected_calibration_sha256':sha(frozen),'reservation_protocol_sha256':sha(reserve/'protocol.json'),
        'reservation_audit_sha256':sha(reserve/'reservation_audit.json'),'weights_sha256':sha(master),'confirmation_used_for_tuning':False,
        'decision':'Report mean CE increase<=log(1.05) and mean token-accuracy loss<=0.01 against FP32 for each frozen native condition. Retain all outcomes; no selection or retuning.',
        'uncertainty':'Normal 1.96*s/sqrt(5) MC intervals conditional on frozen text; additionally 10000 paired bootstrap resamples of five seed indices and sixteen eight-context bank clusters, bootstrap seed20261106. Bootstrap is resampling sensitivity under exchangeable bank clusters, not a distribution-free corpus guarantee.',
        'driver_source_sha256':sha(__file__)}
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Frozen protocol changed; preserve previous run and investigate'
    else:save_json(pp,protocol)
    started=time.monotonic();baseline_rows=[];raw={}
    try:
        for condition in ('FP32','affine_zero_block128_4'):
            if condition!='FP32':
                with torch.no_grad():
                    for module,w in zip(runner.modules,runner.weights):module.weight.copy_(torch.from_numpy(quantize(w,4,'affine_zero',128)))
            result=runner.evaluate(tokens);baseline_rows.append({'condition':condition,**result['scores']})
            raw[condition+'_nll']=result['nll'];raw[condition+'_predictions']=result['predictions']
            print(f'Frozen confirmation baseline {condition}: {result["scores"]}',flush=True)
    finally:runner.restore()
    write_csv(out/'baseline_metrics.csv',baseline_rows);np.savez_compressed(out/'baseline_tokens.npz',**raw)
    rows=[];prefixes=[]
    for condition,worker_module in [('native_float','research.thrml_spw.block_grouped_worker'),('native_finite8','research.thrml_spw.block_control_worker')]:
        log=(out/f'{condition}_worker.log').open('a')
        worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m',worker_module],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
        def reply():
            line=worker.stdout.readline()
            if not line:raise RuntimeError(f'Worker exited {worker.poll()}; see {condition}_worker.log')
            response=json.loads(line)
            if response['status']=='error':raise RuntimeError(str(response))
            return response
        try:
            assert reply()['status']=='ready'
            for seed in protocol['sampling_seeds']:
                directory=out/'runs'/f'{condition}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
                replacements={};runraw={};stages=[];checks=0
                for stage in range(4):
                    result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks']
                    runraw[f'prefix{stage}_nll']=result['nll'];runraw[f'prefix{stage}_predictions']=result['predictions']
                    prefixes.append({'condition':condition,'sampling_seed':seed,'converted_prefix':stage,**result['scores']})
                    xp,rp,yp=[directory/f'stage{stage}_{name}.npy' for name in ('inputs','conditional_reference','native_output')]
                    save_array(xp,result['inputs']);save_array(rp,result['outputs'])
                    request={'weights':str(master),'inputs':str(xp),'reference':str(rp),'output':str(yp),'stage':stage,'samples':256,'seed':seed,'domain':960730,'contexts_per_batch':8}
                    if condition=='native_finite8':request.update(field_bits=bits,field_bound=4.)
                    worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok'
                    meta=json.loads(Path(response['metadata']).read_text());assert meta['output_sha256']==sha(yp)
                    replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'))
                    stages.append({'stage':stage,'inputs':str(xp),'reference':str(rp),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(directory/f'stage{k}_native_output.npy') for k in range(stage)]})
                    print(f'Confirmation {condition} seed={seed} stage={stage} elapsed={time.monotonic()-started:.1f}s',flush=True)
                final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==1280
                runraw['prefix4_nll']=final['nll'];runraw['prefix4_predictions']=final['predictions'];np.savez_compressed(directory/'prefix_tokens.npz',**runraw)
                prefixes.append({'condition':condition,'sampling_seed':seed,'converted_prefix':4,**final['scores']})
                record={'status':'complete','split':'confirmation','condition':condition,'sampling_seed':seed,'samples':256,'field_bits':bits if condition=='native_finite8' else None,
                    'stages':stages,'scores':final['scores'],'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
                save_json(directory/'run.json',record);rows.append({k:record[k] for k in ('condition','sampling_seed','samples','field_bits','free_node_updates','exact_prefix_input_checks')}|final['scores'])
                write_csv(out/'native_metrics.csv',rows);write_csv(out/'prefix_metrics.csv',prefixes)
                print(f'Complete frozen {condition} seed={seed}: {final["scores"]}',flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
            worker.stdin.close();worker.stdout.close();log.close()
    reference=baseline_rows[0];summaries=[]
    for condition in ('native_float','native_finite8'):
        group=[r for r in rows if r['condition']==condition];ce=np.array([r['cross_entropy'] for r in group]);mean=float(ce.mean());margin=float(1.96*ce.std(ddof=1)/np.sqrt(5));acc=float(np.mean([r['token_accuracy'] for r in group]));delta=mean-reference['cross_entropy']
        summaries.append({'condition':condition,'seeds':5,'cross_entropy_mean':mean,'cross_entropy_ci95_margin':margin,'perplexity':float(np.exp(mean)),
            'relative_perplexity':float(np.exp(delta)),'relative_perplexity_lower':float(np.exp(delta-margin)),'relative_perplexity_upper':float(np.exp(delta+margin)),
            'token_accuracy_mean':acc,'mean_quality_tolerance_passed':bool(delta<=np.log(1.05) and reference['token_accuracy']-acc<=.01),'free_node_updates':group[0]['free_node_updates']})
    write_csv(out/'summary.csv',summaries)
    save_json(out/'manifest.json',{'config':protocol,'run_count':len(rows),'summary':summaries,'elapsed_seconds':time.monotonic()-started,'all_student_inputs_regenerated_and_checked':True,'no_confirmation_tuning':True})
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
