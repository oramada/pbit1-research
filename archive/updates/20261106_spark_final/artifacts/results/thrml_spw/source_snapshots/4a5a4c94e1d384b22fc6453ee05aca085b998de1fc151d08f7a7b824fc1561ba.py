"""Execute the already-frozen quarter-model confirmation without tuning."""
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


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    root=Path('results/thrml_spw');prep=root/'quarter_confirmation_preparation';source=root/'transformer_quarter';out=root/'quarter_confirmation';out.mkdir(exist_ok=True)
    frozen_path=prep/'frozen_evaluation.json';p=json.loads(frozen_path.read_text())
    assert json.loads((source/'evidence_audit.json').read_text())['passed'],'Development evidence must be audited first'
    assert sha(source/'selection.json')==p['selection_sha256'] and sha(source/'protocol.json')==p['development_protocol_sha256']
    assert sha(prep/'protocol.json')==p['reservation_protocol_sha256'] and sha(prep/'reservation_audit.json')==p['reservation_audit_sha256']
    assert sha(p['input_data_path'])==p['input_export_sha256'];tokens=np.load(p['input_data_path'])['tokens'];assert tokens.shape==(128,65)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order']);count=len(runner.modules);assert count==12
    master=out/'weights.npz'
    if not master.exists():shutil.copyfile(source/'weights.npz',master)
    assert sha(master)==sha(source/'weights.npz')
    protocol={**p,'frozen_evaluation_sha256':sha(frozen_path),'weights_sha256':sha(master),'driver_source_sha256':sha(__file__),
        'runner_source_sha256':sha(Path(__file__).with_name('sequential_transformer.py'))}
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Do not change an existing confirmation protocol'
    else:save_json(pp,protocol)
    t=p['selected_samples'];started=time.monotonic();base_rows=[];raw={}
    for condition in p['conditions']:
        if condition=='native_finite8':continue
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for module,w in zip(runner.modules,runner.weights):
                        q=expected_weights(w,8,4.) if condition=='programmed_mean8' else quantize(w,4 if condition=='affine4_block128' else 8,'affine_zero',128)
                        module.weight.copy_(torch.from_numpy(q))
            result=runner.evaluate(tokens);base_rows.append({'condition':condition,**result['scores']})
            raw[condition+'_nll']=result['nll'];raw[condition+'_predictions']=result['predictions'];print(f'Frozen quarter confirmation {condition}: {result["scores"]}',flush=True)
        finally:runner.restore()
    write_csv(out/'baseline_metrics.csv',base_rows);np.savez_compressed(out/'baseline_tokens.npz',**raw)
    rows=[];prefixes=[];log=(out/'worker.log').open('a')
    worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.block_cached_control_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}, see worker.log')
        value=json.loads(line)
        if value['status']=='error':raise RuntimeError(str(value))
        return value
    try:
        assert reply()['status']=='ready'
        for seed in p['sampling_seeds']:
            directory=out/'runs'/f'confirmation_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            replacements={};stages=[];raw={};checks=0
            for stage in range(count):
                capture=runner.evaluate(tokens,replacements,capture=stage);checks+=capture['input_checks']
                raw[f'prefix{stage}_nll']=capture['nll'];raw[f'prefix{stage}_predictions']=capture['predictions'];prefixes.append({'sampling_seed':seed,'converted_prefix':stage,**capture['scores']})
                xp,rp,yp=[directory/f'stage{stage}_{name}.npy' for name in ['inputs','conditional_reference','native_output']]
                save_array(xp,capture['inputs']);save_array(rp,capture['outputs'])
                request={'weights':str(master),'inputs':str(xp),'reference':str(rp),'output':str(yp),'stage':stage,'samples':t,'seed':seed,'domain':p['confirmation_domain'],'contexts_per_batch':8,'field_bits':8,'field_bound':4.}
                worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok'
                meta=json.loads(Path(response['metadata']).read_text());assert meta['output_sha256']==sha(yp)
                replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'));stages.append({'stage':stage,'inputs':str(xp),'reference':str(rp),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(directory/f'stage{k}_native_output.npy') for k in range(stage)]})
                print(f'Quarter confirmation seed{seed} stage{stage}/{count-1} elapsed={time.monotonic()-started:.1f}s',flush=True)
            final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==128*78
            raw['prefix12_nll']=final['nll'];raw['prefix12_predictions']=final['predictions'];np.savez_compressed(directory/'prefix_tokens.npz',**raw);prefixes.append({'sampling_seed':seed,'converted_prefix':12,**final['scores']})
            record={'status':'complete','split':'confirmation','samples':t,'sampling_seed':seed,'scores':final['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
            save_json(directory/'run.json',record);rows.append({k:record[k] for k in ['samples','sampling_seed','exact_prefix_input_checks','free_node_updates']}|final['scores']);write_csv(out/'native_metrics.csv',rows);write_csv(out/'prefix_metrics.csv',prefixes)
            print(f'Quarter confirmation seed{seed}: {final["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    ce=np.array([r['cross_entropy'] for r in rows]);mean=float(ce.mean());margin=float(1.96*ce.std(ddof=1)/np.sqrt(5));acc=float(np.mean([r['token_accuracy'] for r in rows]));ref=next(r for r in base_rows if r['condition']=='FP32');delta=mean-ref['cross_entropy']
    summary={'samples':t,'cross_entropy_mean':mean,'cross_entropy_ci95_margin':margin,'perplexity':float(np.exp(mean)),'relative_perplexity':float(np.exp(delta)),
        'relative_perplexity_lower':float(np.exp(delta-margin)),'relative_perplexity_upper':float(np.exp(delta+margin)),'token_accuracy_mean':acc,'mean_quality_tolerance_passed':bool(delta<=np.log(1.05) and ref['token_accuracy']-acc<=.01),'free_node_updates':rows[0]['free_node_updates']}
    write_csv(out/'summary.csv',[summary]);save_json(out/'manifest.json',{'config':protocol,'summary':summary,'run_count':len(rows),'no_confirmation_tuning':True,'all_student_inputs_regenerated_and_checked':True,'elapsed_seconds':time.monotonic()-started});print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
