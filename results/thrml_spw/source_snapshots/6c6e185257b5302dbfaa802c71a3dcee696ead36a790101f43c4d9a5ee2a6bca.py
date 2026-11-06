"""Predeclared native controller-bit search on a sequential four-projection block."""
import json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .transformer_block_gpu import BlockRunner,TARGETS,sha,save_array,save_json,write_csv
from .gpu_environment import cuda_environment


def expected_weights(w,bits,bound):
    lower=np.repeat(w.reshape(-1,128,w.shape[1]).min(1),128,axis=0)
    span=np.repeat(w.reshape(-1,128,w.shape[1]).max(1),128,axis=0)-lower
    p=np.zeros_like(w);np.divide(w-lower,span,out=p,where=span!=0)
    p=np.clip(p,0,1).astype(np.float64);free=(p>0)&(p<1)
    requested=((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32).astype(np.float64)
    codes=np.rint((np.clip(requested,-bound,bound)+bound)*((1<<bits)-1)/(2*bound))
    b=(-bound+codes*2*bound/((1<<bits)-1)).astype(np.float32).astype(np.float64)
    p[free]=np.exp(-np.logaddexp(0,-2*b))
    return (lower+span*p).astype(np.float32)


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    root=Path('results/thrml_spw');out=root/'transformer_control';out.mkdir(exist_ok=True)
    source=root/'transformer_block0_group128';base=json.loads((source/'protocol.json').read_text())
    data=np.load('data/thrml_spw/transformer_projection/export.npz')
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{base['revision']}"
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval()
    runner=BlockRunner(model);master=out/'weights.npz'
    if not master.exists():shutil.copyfile(source/'weights.npz',master)
    assert sha(master)==sha(source/'weights.npz')
    protocol={**base,'field_bit_search':[4,6,8,10,12],'field_bound':4.,'samples':256,
        'selection':'Five native calibration seeds per ascending bit count, fixed T256 and field bound4. Stop at first mean CE increase<=log(1.05) and mean token-accuracy loss<=0.01. If none pass report failure. Test chosen bits and predeclared4 with five seeds; no test-based selection.',
        'control_model':'Uniform dimensionless Ising bias levels, float32 containers, explicit endpoint clamps. Not measured hardware voltage or physical DAC.',
        'scope':'Development text previously inspected; all four block0 matrix projections (8.64% model), grouped128 affine encoding, five MC seeds; not full-model/hardware validation.',
        'driver_source_sha256':sha(__file__),'worker_source_sha256':sha(Path(__file__).with_name('block_control_worker.py'))}
    save_json(out/'protocol.json',protocol)
    baselines={};baseline_rows=[];raw={}
    for split in ('validation','test'):
        result=runner.evaluate(data[split+'_tokens'])
        np.testing.assert_array_equal(result['nll'],data[split+'_reference_token_nll'])
        baselines[split]=result['scores'];baseline_rows.append({'split':split,'condition':'FP32',**result['scores']})
        raw[split+'_FP32_nll']=result['nll'];raw[split+'_FP32_predictions']=result['predictions']
    write_csv(out/'baseline_metrics.csv',baseline_rows);np.savez_compressed(out/'baseline_tokens.npz',**raw)
    log=(out/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.block_control_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    rows=[];prefixes=[];profiles=[];started=time.monotonic()
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}, see worker.log')
        value=json.loads(line)
        if value['status']=='error':raise RuntimeError(str(value))
        return value
    try:
        assert reply()['status']=='ready'
        def run(split,bits,seed):
            directory=out/'runs'/f'{split}_B{bits}_T256_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            tokens=data[split+'_tokens'];replacements={};raw={};stages=[];checks=0
            for stage in range(4):
                result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks']
                raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions']
                prefixes.append({'split':split,'field_bits':bits,'samples':256,'sampling_seed':seed,'converted_prefix':stage,**result['scores']})
                xp,rp,yp=[directory/f'stage{stage}_{name}.npy' for name in ('inputs','conditional_reference','native_output')]
                save_array(xp,result['inputs']);save_array(rp,result['outputs'])
                request={'weights':str(master),'inputs':str(xp),'reference':str(rp),'output':str(yp),'stage':stage,'samples':256,'seed':seed,
                    'domain':base['calibration_domain'] if split=='validation' else base['test_domain'],'contexts_per_batch':8,'field_bits':bits,'field_bound':4.}
                worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok'
                metadata=json.loads(Path(response['metadata']).read_text());assert metadata['output_sha256']==sha(yp)
                replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'))
                stages.append({'stage':stage,'inputs':str(xp),'reference':str(rp),'output':str(yp),'metadata':response['metadata'],
                    'prior_native_output_sha256':[sha(directory/f'stage{k}_native_output.npy') for k in range(stage)]})
                print(f'Native {split} bits={bits} seed={seed} stage={stage} complete elapsed={time.monotonic()-started:.1f}s',flush=True)
            final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==len(tokens)*10
            raw['prefix4_nll']=final['nll'];raw['prefix4_predictions']=final['predictions']
            np.savez_compressed(directory/'prefix_tokens.npz',**raw)
            prefixes.append({'split':split,'field_bits':bits,'samples':256,'sampling_seed':seed,'converted_prefix':4,**final['scores']})
            record={'status':'complete','split':split,'field_bits':bits,'samples':256,'sampling_seed':seed,'scores':final['scores'],'stages':stages,
                'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
            save_json(directory/'run.json',record)
            rows.append({k:record[k] for k in ('split','field_bits','samples','sampling_seed','free_node_updates','exact_prefix_input_checks')}|final['scores'])
            write_csv(out/'native_metrics.csv',rows);write_csv(out/'prefix_metrics.csv',prefixes)
            print(f"Complete {split} bits={bits} seed={seed} perplexity={final['scores']['perplexity']:.3f}",flush=True)
        selected=None
        for bits in protocol['field_bit_search']:
            for seed in range(5):run('validation',bits,seed)
            group=[r for r in rows if r['split']=='validation' and r['field_bits']==bits]
            ce=float(np.mean([r['cross_entropy'] for r in group]));acc=float(np.mean([r['token_accuracy'] for r in group]))
            passed=bool(ce-baselines['validation']['cross_entropy']<=np.log(1.05) and baselines['validation']['token_accuracy']-acc<=.01)
            profile={'field_bits':bits,'cross_entropy_mean':ce,'relative_perplexity':float(np.exp(ce-baselines['validation']['cross_entropy'])),'token_accuracy_mean':acc,'passes':passed}
            profiles.append(profile);save_json(out/'calibration_progress.json',{'profiles':profiles,'test_labels_used_in_selection':False});print(profile,flush=True)
            if passed:selected=bits;break
        chosen=selected if selected is not None else protocol['field_bit_search'][-1]
        save_json(out/'selection.json',{'selected_bits':chosen,'gate_passed':selected is not None,'profiles':profiles,'test_labels_used_in_selection':False})
        for bits in sorted({4,chosen}):
            for seed in range(5):run('test',bits,seed)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    # Programmed expected-weight controls isolate persistent controller error from sampling.
    expected_rows=[];expected_raw={}
    try:
        for bits in protocol['field_bit_search']:
            with torch.no_grad():
                for module,w in zip(runner.modules,runner.weights):module.weight.copy_(torch.from_numpy(expected_weights(w,bits,4.)))
            for split in ('validation','test'):
                result=runner.evaluate(data[split+'_tokens']);expected_rows.append({'split':split,'field_bits':bits,**result['scores']})
                expected_raw[f'{split}_B{bits}_nll']=result['nll'];expected_raw[f'{split}_B{bits}_predictions']=result['predictions']
    finally:runner.restore()
    write_csv(out/'expected_metrics.csv',expected_rows);np.savez_compressed(out/'expected_tokens.npz',**expected_raw)
    summaries=[]
    for split,bits in sorted({(r['split'],r['field_bits']) for r in rows}):
        group=[r for r in rows if r['split']==split and r['field_bits']==bits];ce=np.array([r['cross_entropy'] for r in group]);mean=float(ce.mean());margin=float(1.96*ce.std(ddof=1)/np.sqrt(5));ref=baselines[split]['cross_entropy']
        summaries.append({'split':split,'field_bits':bits,'samples':256,'cross_entropy_mean':mean,'cross_entropy_ci95_margin':margin,'perplexity':float(np.exp(mean)),
            'relative_perplexity':float(np.exp(mean-ref)),'relative_perplexity_lower':float(np.exp(mean-margin-ref)),'relative_perplexity_upper':float(np.exp(mean+margin-ref)),
            'token_accuracy_mean':float(np.mean([r['token_accuracy'] for r in group])),'free_node_updates':group[0]['free_node_updates']})
    write_csv(out/'summary.csv',summaries)
    save_json(out/'manifest.json',{'config':protocol,'run_count':len(rows),'selection':json.loads((out/'selection.json').read_text()),'summary':summaries,'weights_sha256':sha(master),'elapsed_seconds':time.monotonic()-started,'all_student_inputs_regenerated_and_checked':True})
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
