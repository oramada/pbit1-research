"""H100 execution; explicit user authorization received before deployment."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/code')
app = modal.App('spw-h100-heldout-v1')
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


@app.function(image=image,gpu='H100!',cpu=8,memory=65536,ephemeral_disk=524288,
              volumes={'/workspace':volume},timeout=86400,retries=0,max_containers=4,scaledown_window=10)
def case(seed: int):
    import os,subprocess
    setup_workspace();volume.reload()
    output=Path('/workspace/cloud_holdout_logs_v1');output.mkdir(exist_ok=True)
    with (output/f'{seed}.log').open('x') as log:
        for name in ['holdout_native_case','verify_holdout_native_case']:
            result=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.'+name,'--seed',str(seed)],
                cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            volume.commit()
            assert result.returncode==0,{'seed':seed,'module':name,'exit_code':result.returncode}
    return {'seed':seed,'status':'heldout_case_and_independent_audit_complete'}

@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=86400,retries=0,max_containers=1)
def campaign():
    import time,hashlib,concurrent.futures,subprocess,os
    setup_workspace()
    path=Path('/workspace/holdout_campaign_v1.json')
    state={'status':'waiting_complete_validation_and_frozen_execution_protocol','completed':[]}
    def save():
        path.write_text(json.dumps(state,indent=2)+'\n');volume.commit();print(json.dumps(state),flush=True)
    save();deadline=time.time()+24*3600
    while True:
        volume.reload()
        ready=Path('/workspace/holdout_assets_v1.json')
        if ready.exists():break
        assert time.time()<deadline,'No eligible, completely frozen holdout protocol after24hours'
        time.sleep(30)
    m=json.loads(ready.read_text())
    for name,rec in m['files'].items():
        p=Path('/workspace')/name
        assert p.stat().st_size==rec['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],name
    p=json.loads(Path('/workspace/results/thrml_spw/lfm3_1p7b/holdout_native_protocol.json').read_text())
    state['status']='first_H100_heldout_seed';save()
    state['completed'].append(case.remote(p['seeds'][0]));save()
    state['status']='four_parallel_H100_heldout_seeds';save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(case.remote,seed) for seed in p['seeds'][1:]]
        for future in concurrent.futures.as_completed(futures):
            state['completed'].append(future.result());save()
    volume.reload();state['status']='independent_paired_holdout_statistics';save()
    with Path('/workspace/cloud_holdout_logs_v1/statistics.log').open('x') as log:
        for module in ['holdout_native_statistics','verify_holdout_native_statistics']:
            r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.'+module],
                cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            volume.commit();assert r.returncode==0,{'module':module,'exit_code':r.returncode}
    state['status']='five_heldout_seeds_and_independent_paired_statistics_complete';save()
    return state

@app.local_entrypoint()
def launch():
    print(json.dumps(campaign.remote()))
