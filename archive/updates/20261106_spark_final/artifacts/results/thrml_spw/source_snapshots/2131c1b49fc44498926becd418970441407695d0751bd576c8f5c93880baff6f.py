"""H100 execution; explicit user authorization received before deployment."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/code')
app = modal.App('spw-h100-experiments-v6')
volume = modal.Volume.from_name('spw-h100-validation-v1', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'libgomp1')
    .run_commands(
        'python -m venv /opt/torch',
        '/opt/torch/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 numpy==2.5.3 transformers==5.21.0 huggingface_hub==2.3.0 safetensors==0.8.0',
        'python -m venv /opt/thrml',
        '/opt/thrml/bin/pip install "jax[cuda13]==0.12.0.1" numpy==2.5.4 equinox==0.13.8 jaxtyping==0.3.11 "thrml @ git+https://github.com/extropic-ai/thrml.git@2d54cd8404df85bb65703d7e8b4e84e48c1b72a9"',
    )
    .env({'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4',
          'XLA_FLAGS': '--xla_gpu_autotune_level=0',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
          'JAX_DEFAULT_MATMUL_PRECISION': 'highest', 'PYTHONPATH': '/code'})
    .add_local_dir(ROOT/'research', '/code/research', copy=True, ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'cloud', '/code/cloud', copy=True, ignore=['**/__pycache__/**'])
)


def setup_workspace():
    import os
    root=Path('/workspace')
    for name,target in [('.venv','/opt/torch'),('.venv-thrml-gpu','/opt/thrml'),('research','/code/research'),('cloud','/code/cloud')]:
        path=root/name
        if os.path.lexists(path):
            assert path.is_symlink() and os.readlink(path)==target
        else:path.symlink_to(target)
    if not os.path.lexists('/code/.venv'):Path('/code/.venv').symlink_to('/opt/torch')


@app.function(image=image,gpu='H100!',cpu=8,memory=65536,ephemeral_disk=524288,volumes={'/workspace':volume},timeout=14400,retries=0,max_containers=4,scaledown_window=10)
def run_case(module: str, arguments: list[str], label: str):
    """Only versioned cloud runners may write new case namespaces."""
    import os
    import subprocess
    assert module.startswith('cloud.modal_h100_v1.')
    setup_workspace()
    volume.reload()
    logdir = Path('/workspace/cloud_execution_logs_v1')
    logdir.mkdir(exist_ok=True)
    path = logdir/(label+'.log')
    with path.open('x') as log:
        code = subprocess.call(['/opt/torch/bin/python', '-m', module, *arguments], cwd='/workspace',
            env={**os.environ, 'PYTHONPATH':'/code'}, stdout=log, stderr=log)
    volume.commit()
    return {'label':label,'exit_code':code,'log':str(path)}


@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=7200,retries=0,max_containers=1)
def prepare_and_audit_existing_bridge():
    import hashlib,numpy as np,subprocess,os,datetime
    setup_workspace();volume.reload()
    root=Path('/workspace/results/thrml_spw/lfm3_1p7b')
    case=root/'modal_h100_word_v2/C1024/K8/mean/202610350'
    gold=json.loads(Path('/workspace/h100_parity_reference_v1.json').read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes={}
    for batch in range(4):
        with np.load(case/f'batch{batch}.npz') as arrays:
            for stage in ['BF16','same_fit_BF16','nominal_controls','native']:
                h=hashlib.sha256(arrays[stage].tobytes()).hexdigest()
                if stage in ['nominal_controls','native']:assert h==gold['array_sha256'][f'{batch}/{stage}']
                else:hashes[f'{batch}/{stage}']=h
    reference={'status':'H100_backend_specific_BF16_controls_frozen_before_continuation',
        'array_sha256':hashes,'source_bridge_results_sha256':sha(case/'results.json'),
        'original_failed_parity_sha256':sha(case/'cross_backend_parity.json')}
    rp=root/'h100_reference_controls_v3.json'
    with rp.open('x') as f:json.dump(reference,f,indent=2)
    amendment={'status':'explicit_user_authorized_backend_control_amendment_before_continuation',
        'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_producer_protocol_sha256':sha(root/'modal_h100_word_protocol_v2.json'),
        'H100_reference_sha256':sha(rp),
        'new_auditor_sha256':sha('/code/research/thrml_spw/lfm3/verify_modal_h100_word_v3.py'),
        'new_wrapper_sha256':sha('/code/cloud/modal_h100_v1/word_case_v3.py'),
        'orchestrator_sha256':sha('/code/cloud/modal_h100_v6/app.py'),
        'quality_NLL_margin':0.01980262729617973,
        'known_control_shift_same_fit_NLL':0.00005150653649454995,
        'policy':'Use exact H100 master control arrays for H100 cases. Original local BF16 controls and failed bridge retained. Unchanged stochastic producer, seeds, work, and2percentqualitymargin. Backend shift is deterministic, not an estimated variance. Native/nominal bridge arrays match exactly; no full-success claim. Report controls for each backend separately and disclose this post-observation user-authorized amendment.',
        'scope':'Implementation continuation, not a new preregistered discovery or relaxation of neural quality criteria.'}
    with (root/'h100_backend_control_amendment_v3.json').open('x') as f:json.dump(amendment,f,indent=2)
    volume.commit()
    logs=Path('/workspace/cloud_execution_logs_v1')
    with (logs/'existing_bridge_audit_v3.log').open('x') as log:
        r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.verify_modal_h100_word_v3','--arm','mean','--budget','1024','--word-bits','8','--seed','202610350'],cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
    volume.commit();assert r.returncode==0,{'audit_exit_code':r.returncode}
    return {'status':'existing_sampled_bridge_reused_and_independently_audited_against_H100_controls','amendment':amendment}

@app.function(image=image,cpu=4,memory=8192,volumes={'/workspace':volume},timeout=86400,retries=0,max_containers=1)
def campaign():
    import concurrent.futures,time
    setup_workspace();path=Path('/workspace/campaign_v3.json')
    state={'status':'preparing_H100_controls_and_auditing_existing_bridge','completed':[],'failed':[]}
    def save():
        path.write_text(json.dumps(state,indent=2)+'\n');volume.commit();print(json.dumps(state),flush=True)
    save();state['bridge']=prepare_and_audit_existing_bridge.remote();save()
    specs=[]
    for seed in [202610352,202610353,202610354]:specs.append((1024,8,seed,None))
    for bits in [4,8]:
        specs.append((4096,bits,202610420,None))
        for seed in range(202610421,202610425):specs.append((4096,bits,seed,f'C4096_K{bits}_202610420'))
    def execute(spec):
        budget,bits,seed,_=spec;label=f'C{budget}_K{bits}_{seed}'
        return run_case.remote('cloud.modal_h100_v1.word_case_v3',['--budget',str(budget),'--bits',str(bits),'--seed',str(seed)],label+'_v3')
    completed=set();deadline=time.time()+86400
    for phase in ['C1024_C4096','C8192']:
        if phase=='C8192':
            state['status']='waiting_C8192_complete_reference_metadata';save()
            while True:
                volume.reload()
                if Path('/workspace/C8192_assets_v1.json').exists():break
                assert time.time()<deadline;time.sleep(30)
            for bits in [4,8]:
                specs.append((8192,bits,202610600,None))
                for seed in range(202610601,202610605):specs.append((8192,bits,seed,f'C8192_K{bits}_202610600'))
        state['status']='parallel_H100_'+phase;save();active={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            while specs or active:
                for spec in list(specs):
                    if len(active)>=4:break
                    if spec[3] is None or spec[3] in completed:
                        active[pool.submit(execute,spec)]=f'C{spec[0]}_K{spec[1]}_{spec[2]}';specs.remove(spec)
                assert active,'Unmet first-seed audit dependency'
                state['running']=list(active.values());save()
                done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    label=active.pop(future);r=future.result()
                    if r['exit_code']!=0:
                        state['failed'].append(r);state['status']='sampled_case_failed_preserved_no_retry';save();raise RuntimeError(str(r))
                    completed.add(label);state['completed'].append(label);save()
    state['running']=[];state['status']='all_H100_comparison_cases_and_backend_specific_audits_complete_statistics_pending';save();return state

@app.local_entrypoint()
def launch():print(json.dumps(campaign.remote()))
