"""Diagnostic control-error separation on the already consumed Transformer study."""
import json,subprocess,time,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .transformer_block_gpu import sha,save_json,save_array,write_csv
from .sequential_transformer import SequentialRunner
from .calibrated_transformer_controls import device,decode
from .transformer_floor_math import fields,moments,METHODS
from .gpu_environment import cuda_environment
OLD=Path('results/thrml_spw/calibrated_transformer');OUT=Path('results/thrml_spw/transformer_calibration_floor')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    assert json.loads((OLD/'evidence_audit.json').read_text())['passed'];design=json.loads((OLD/'protocol.json').read_text())
    OUT.mkdir(exist_ok=True);(OUT/'mean_runs').mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    sources=['transformer_calibration_floor.py','transformer_floor_worker.py','transformer_floor_math.py','calibrated_transformer_controls.py','calibrated_transformer_native.py','sequential_transformer.py','transformer_block_gpu.py']
    protocol={'schema':'spw-transformer-calibration-floor-diagnostic-v1','revision':design['revision'],'targets_in_forward_order':design['targets_in_forward_order'],
        'source_study':str(OLD),'source_manifest_sha256':sha(OLD/'manifest.json'),'source_evidence_audit_sha256':sha(OLD/'evidence_audit.json'),'source_native_audit_sha256':sha(OLD/'native_path_audit.json'),
        'tokens_path':str(OLD/'tokens.npz'),'tokens_sha256':sha(OLD/'tokens.npz'),'control_artifacts':design['control_artifacts'],'converted_weights':21233664,'model_unique_parameters':81912576,
        'mean_conditions':list(METHODS),'mean_runs':16,'native_condition':'oracle8','native_runs':5,'device_seeds':list(range(5)),'native_samples':1024,'contexts_per_bank':8,'windows':64,'targets':4096,
        'counterfactuals':'ideal8=nominal8bitfields; measured8=oldmeasuredP4096 commands realizedonthesamephysicaldevice; oracle8=truegain/offset inverse clipped[-4,4] and8bitrounded; oracle_continuous_bound4=trueinverseclipped[-4,4] withoutgridrounding. Actualfields emittedF32; analyticmarginals sigmoid(2field) computedF64 andNNweights thenF32.',
        'device_and_keys':'Identicaloldprofiles SeedSequence([2026114600,stage,seed]) and native1140730foldseed,stage,bank. Fivepairedcounterfactuals, notnewrandomreplicates. No native probes: old measuredcalcounts reused byhash; trueparams suppliedonlytoexplicitknown-parameterdiagnostics.',
        'moments':'Exactweight-error telescope measured-ideal=(measured-oracle8)+(oracle8-oraclecontinuousbound4)+(oraclecontinuousbound4-ideal), includingallthree signedcross terms inSSE. Thisisnotanadditiveindependent-noise or task-loss attribution. Report alltarget-field/matrixhashes andclippedinversefractions.',
        'decision':'Executeall16deterministicmeanforwardsand5nativeoracleforwards. Primary diagnostic contrast known-parameteroracle8 versuspreviousmeasured8, withcontrollerbits/bounds/nativeT/devicepaths/MCkeys/text fixed. Continuous-range/grid diagnosticseparate. No newqualitygate, training, test-drivenextension orcandidate-selection.',
        'scope':'Mechanisticfollow-uponalreadyconsumed64WikiText2windows andonepretrainedmodel, firstthreeblocks25.92%. Mean-weight network isnotexpectationofnoisy nonlinear network. Oracleknowsunobservabletrueparams: notdeployablecalibration, notnewheld-outconfirmation, notrescueofoldfailed5%gate. No physical orfull-modeladvantageclaim.',
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in sources}}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:save_json(pp,protocol)
    archive=Path('results/thrml_spw/source_snapshots');archive.mkdir(exist_ok=True)
    for n,h in protocol['source_sha256'].items():
        p=archive/(h+'.py');data=Path(n).read_bytes()
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
    tokens=np.load(OLD/'tokens.npz')['tokens'];oldbase=np.load(OLD/'baseline_tokens.npz');reference=float(oldbase['FP32_nll'].astype(float).mean())
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{design['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),design['targets_in_forward_order'])
    meanrows=[];diagnostics=[];start=time.monotonic()
    for seed in range(5):
        weights={method:[] for method in METHODS};hashes={method:[] for method in METHODS}
        for stage,c in enumerate(design['control_artifacts']):
            assert sha(c['path'])==c['sha256'];a=np.load(c['path']);shape=a['codes'].shape;nominal=decode(a['codes']).reshape(-1);gain,offset=device(stage,seed,nominal.size)
            cp=OLD/'calibration'/f'stage{stage}_seed{seed}.npz';cm=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==cm['archive_sha256'];measured=np.load(cp)['corrected_codes']
            actual,oracle,inverse=fields(nominal,gain,offset,measured)
            oldmeta=json.loads((OLD/'runs'/f'cal4096_seed{seed}'/f'stage{stage}_native_output.json').read_text());assert digest(actual['measured8'])==oldmeta['field_sha256']
            span=np.repeat(a['span'],128,0);lower=np.repeat(a['lower'],128,0);q={k:1/(1+np.exp(-2*f.astype(float))) for k,f in actual.items()}
            row={'stage':stage,'seed':seed,'calibration_archive_sha256':sha(cp),'true_inverse_bound_clipped':int(np.count_nonzero(np.abs(inverse)>4)),'field_sha256':{k:digest(v) for k,v in actual.items()},'oracle_code_sha256':digest(oracle),
                'moments':moments(q,span.reshape(-1)), 'mean_matrix_sha256':{},'target_MSE':{}}
            for method in METHODS:
                full=lower+span*q[method].reshape(shape);value=full.astype(np.float32);row['target_MSE'][method]=float(np.mean((full-runner.weights[stage].astype(float))**2))
                row['mean_matrix_sha256'][method]=digest(value);weights[method].append(value);hashes[method].append(digest(value))
            diagnostics.append(row)
        for method in METHODS:
            if method=='ideal8' and seed:continue
            stem=f'{method}_seed{seed}';path=OUT/'mean_runs'/f'{stem}.npz';jp=path.with_suffix('.json')
            if jp.exists():
                row=json.loads(jp.read_text());assert sha(path)==row['archive_sha256'] and hashes[method]==row['mean_matrix_sha256']
            else:
                try:
                    with torch.no_grad():
                        for module,weight in zip(runner.modules,weights[method]):module.weight.copy_(torch.from_numpy(weight))
                    result=runner.evaluate(tokens)
                    if method=='ideal8':
                        np.testing.assert_array_equal(result['nll'],oldbase['allfree_programmed_mean8_nll']);np.testing.assert_array_equal(result['predictions'],oldbase['allfree_programmed_mean8_predictions'])
                    np.savez_compressed(path,nll=result['nll'],predictions=result['predictions'])
                    row={'condition':method,'device_seed':seed,**result['scores'],'relative_perplexity_to_FP32':float(np.exp(result['scores']['cross_entropy']-reference)),
                        'archive_sha256':sha(path),'mean_matrix_sha256':hashes[method],'native_free_evaluations':0,'scope':'Deterministic analytic-programmed-mean diagnostic, not expectationof sampledReLU/Transformer andnot nativeinference.'}
                    save_json(jp,row)
                finally:runner.restore()
            meanrows.append({k:v for k,v in row.items() if k not in ['mean_matrix_sha256','scope']})
            print(f'Mean diagnostic {stem}: PPL={row["perplexity"]:.4f}, elapsed={time.monotonic()-start:.1f}s',flush=True)
        del weights
    assert len(meanrows)==16 and len(diagnostics)==60
    save_json(OUT/'weight_diagnostics.json',diagnostics);write_csv(OUT/'mean_metrics.csv',meanrows)
    log=(OUT/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.transformer_floor_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker ended{worker.poll()}')
        value=json.loads(line)
        if value['status']=='error':raise RuntimeError(str(value))
        return value
    rows=[];prefix=[]
    try:
        assert reply()['status']=='ready'
        for seed in range(5):
            folder=OUT/'runs'/f'oracle8_seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
            if rp.exists():
                r=json.loads(rp.read_text());assert r['status']=='complete' and r['protocol_sha256']==sha(pp)
                rows.append({'condition':'oracle8','sampling_seed':seed,**r['scores'],'free_node_updates':r['free_node_updates']});continue
            replacements={};raw={};stages=[];checks=0
            for stage in range(12):
                result=runner.evaluate(tokens,replacements,capture=stage);checks+=result['input_checks'];raw[f'prefix{stage}_nll']=result['nll'];raw[f'prefix{stage}_predictions']=result['predictions']
                if stage==0:np.testing.assert_array_equal(result['nll'],oldbase['FP32_nll'])
                xp,yp=[folder/f'stage{stage}_{n}.npy' for n in ['inputs','native_output']];save_array(xp,result['inputs'])
                request={'condition':'oracle8','stage':stage,'seed':seed,'samples':1024,'contexts_per_bank':8,'controls':design['control_artifacts'][stage]['path'],'inputs':str(xp),'output':str(yp)}
                worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text())
                assert sha(yp)==meta['output_sha256'] and meta['field_sha256']==next(d for d in diagnostics if d['stage']==stage and d['seed']==seed)['field_sha256']['oracle8']
                replacements[stage]=(np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r'));stages.append({'stage':stage,'inputs':str(xp),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]})
                print(f'Native known-parameter oracle seed{seed} stage{stage}/11 elapsed={time.monotonic()-start:.1f}s',flush=True)
            final=runner.evaluate(tokens,replacements);checks+=final['input_checks'];assert checks==64*78;raw['prefix12_nll']=final['nll'];raw['prefix12_predictions']=final['predictions'];np.savez_compressed(folder/'prefix_tokens.npz',**raw)
            r={'status':'complete','split':'diagnostic','condition':'oracle8','sampling_seed':seed,'protocol_sha256':sha(pp),'scores':final['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages)}
            save_json(rp,r);rows.append({'condition':'oracle8','sampling_seed':seed,**final['scores'],'free_node_updates':r['free_node_updates']});write_csv(OUT/'native_metrics.csv',rows)
            print(f'Complete native oracle seed{seed}: {final["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    assert len(rows)==5;write_csv(OUT/'native_metrics.csv',rows)
    result={'config':protocol,'mean_forward_runs':16,'native_forward_runs':5,'native_operator_runs':60,'native_probe_runs':0,'native_free_evaluations':sum(r['free_node_updates'] for r in rows),'elapsed_seconds':time.monotonic()-start,'scope':protocol['scope']}
    save_json(OUT/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k!='config'},indent=2))


if __name__=='__main__':main()
