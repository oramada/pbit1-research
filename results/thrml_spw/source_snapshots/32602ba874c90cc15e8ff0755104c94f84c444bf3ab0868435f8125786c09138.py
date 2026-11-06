"""H100 execution; explicit user authorization received before deployment."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/code')
app = modal.App('spw-h100-experiments-v4')
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


@app.function(image=image, gpu='H100!', cpu=8, memory=65536,
              ephemeral_disk=524288, volumes={'/workspace': volume},
              timeout=1800, retries=0, max_containers=1)
def probe():
    import os
    import subprocess
    import time
    begin = time.time()
    root = Path('/workspace')
    output = root/'cloud_probe_v2'
    output.mkdir(exist_ok=False)
    os.symlink('/opt/torch', root/'.venv')
    os.symlink('/opt/thrml', root/'.venv-thrml-gpu')
    os.symlink('/code/research', root/'research')
    script = """import json,torch,importlib.metadata as m
print(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'versions':{n:m.version(n) for n in ['numpy','transformers','huggingface_hub','safetensors']}}))"""
    a = subprocess.run(['/opt/torch/bin/python', '-c', script], capture_output=True, text=True, check=True)
    torch_info = json.loads(a.stdout)
    with (output/'jax.log').open('x') as log:
        b = subprocess.run(['/opt/thrml/bin/python', '-c',
            "import jax,json,importlib.metadata as m;print(json.dumps({'devices':list(map(str,jax.devices())),'versions':{n:m.version(n) for n in ['jax','jaxlib','numpy','thrml']}}))"],
            env={**os.environ, 'JAX_PLATFORMS': 'cuda'}, stdout=subprocess.PIPE, stderr=log, text=True)
    result = {'status':'environment_probe_complete' if b.returncode==0 else 'environment_probe_failed',
        'torch':torch_info,'jax_exit_code':b.returncode,'jax_stdout':b.stdout,
        'elapsed_seconds':time.time()-begin, 'scope':'Environment probe only, no neural scores or equivalence claim.'}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    volume.commit()
    return result


@app.function(image=image, gpu='H100!', cpu=8, memory=65536,
              ephemeral_disk=524288, volumes={'/workspace': volume},
              timeout=14400, retries=0, max_containers=4, scaledown_window=10)
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


@app.local_entrypoint()
def main(mode: str = 'probe'):
    assert mode == 'probe'
    result = probe.remote()
    path = ROOT/'results/thrml_spw/modal_h100_v1'
    path.mkdir(exist_ok=True)
    (path/'environment_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def setup_workspace():
    import os
    root=Path('/workspace')
    for name,target in [('.venv','/opt/torch'),('.venv-thrml-gpu','/opt/thrml'),('research','/code/research'),('cloud','/code/cloud')]:
        path=root/name
        if os.path.lexists(path):
            assert path.is_symlink() and os.readlink(path)==target
        else:path.symlink_to(target)
    if not os.path.lexists('/code/.venv'):Path('/code/.venv').symlink_to('/opt/torch')


@app.function(image=image,cpu=4,memory=8192,volumes={'/workspace':volume},timeout=3600,retries=0,max_containers=1)
def hydrate_checkpoint():
    import subprocess,hashlib
    setup_workspace()
    script="from huggingface_hub import snapshot_download; snapshot_download(repo_id='LiquidAI/LFM3-1.7B',revision='ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',local_dir='/workspace/data/thrml_spw/lfm3_1p7b/checkpoint',allow_patterns=['*.json','*.py','*.safetensors','LICENSE'])"
    subprocess.run(['/opt/torch/bin/python','-c',script],check=True)
    path=Path('/workspace/data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    assert h.hexdigest()=='135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330'
    result={'status':'pinned_checkpoint_hydrated_and_hash_verified','sha256':h.hexdigest(),'bytes':path.stat().st_size}
    Path('/workspace/checkpoint_ready_v1.json').write_text(json.dumps(result)+'\n')
    volume.commit()
    return result


@app.function(image=image,cpu=4,memory=8192,volumes={'/workspace':volume},timeout=86400,retries=0,max_containers=1)
def campaign():
    import time,hashlib,concurrent.futures
    setup_workspace()
    path=Path('/workspace/campaign_v1.json')
    state={'status':'hydrating_checkpoint_and_waiting_fixed_inputs','completed':[],'running':[],'failed':[]}
    def save():
        path.write_text(json.dumps(state,indent=2)+'\n');volume.commit();print(json.dumps(state),flush=True)
    save()
    hydration=hydrate_checkpoint.spawn()
    state['status']='H100_component_check';save()
    result=run_case.remote('cloud.modal_h100_v1.component',[],'component_v1')
    assert result['exit_code']==0,result

    deadline=time.time()+3600
    while True:
        volume.reload()
        if Path('/workspace/asset_manifest_v1.json').exists() and Path('/workspace/cloud_metadata_ready_v1.json').exists():break
        assert time.time()<deadline,'Input upload exceeded one-hour infrastructure deadline'
        time.sleep(15)
    hydration.get()
    volume.reload()
    manifest=json.loads(Path('/workspace/asset_manifest_v1.json').read_text())
    for name,rec in manifest['files'].items():
        target=Path('/workspace')/name
        if not target.exists():
            import gzip,shutil
            compressed=Path('/workspace/compressed_inputs_v1')/(name+'.gz')
            assert compressed.exists(),name
            target.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(compressed,'rb') as src,target.open('xb') as dest:shutil.copyfileobj(src,dest,16*1024*1024)
        h=hashlib.sha256()
        with (Path('/workspace')/name).open('rb') as f:
            for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
        assert h.hexdigest()==rec['sha256'],name
    state['status']='H100_full_model_parity_case';save()
    result=run_case.remote('cloud.modal_h100_v1.word_case',['--budget','1024','--bits','8','--seed','202610350'],'parity_K8_C1024_202610350')
    assert result['exit_code']==0,result
    state['completed'].append('parity_K8_C1024_202610350')
    state['status']='parallel_H100_matched_comparisons';save()
    specs=[]
    for seed in [202610352,202610353,202610354]:specs.append((1024,8,seed,None))
    for bits in [4,8]:
        specs.append((4096,bits,202610420,None))
        for seed in range(202610421,202610425):specs.append((4096,bits,seed,f'C4096_K{bits}_202610420'))
    completed=set();active={}
    def execute(spec):
        budget,bits,seed,_=spec;label=f'C{budget}_K{bits}_{seed}'
        return run_case.remote('cloud.modal_h100_v1.word_case',['--budget',str(budget),'--bits',str(bits),'--seed',str(seed)],label)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while specs or active:
            for spec in list(specs):
                if len(active)>=4:break
                if spec[3] is None or spec[3] in completed:
                    label=f'C{spec[0]}_K{spec[1]}_{spec[2]}'
                    active[pool.submit(execute,spec)]=label;specs.remove(spec)
            assert active,'Unmet comparison dependency'
            state['running']=list(active.values());state['pending']=len(specs);save()
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                label=active.pop(future);result=future.result()
                if result['exit_code']!=0:
                    state['failed'].append(result);state['status']='failed_case_preserved_no_sampling_retry';save()
                    raise RuntimeError(str(result))
                completed.add(label);state['completed'].append(label)
    state['running']=[];state['pending']=0;state['status']='cloud_comparison_cases_and_independent_audits_complete_statistics_pending';save()
    return state


@app.local_entrypoint()
def launch():
    result=campaign.remote()
    print(json.dumps(result))
