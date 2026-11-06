"""H100 execution; explicit user authorization received before deployment."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/code')
app = modal.App('pbit1-broad-preflight-v1')
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

@app.function(image=image,gpu='H100!',cpu=8,memory=32768,ephemeral_disk=524288,volumes={'/workspace':volume},timeout=1800,retries=0,max_containers=1,scaledown_window=10)
def component():
    import os,subprocess
    setup_workspace();volume.reload()
    out=Path('/workspace/cloud_broad_validation_v1');out.mkdir(exist_ok=True)
    with (out/'h100_attention_preflight.log').open('x') as log:
        for name in ['preflight_attention_worker_h100_v2','verify_attention_worker_h100_v2']:
            r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.'+name],cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            volume.commit();assert r.returncode==0,{'module':name,'code':r.returncode}
    return json.loads(Path('/workspace/results/thrml_spw/lfm3_1p7b/attention_worker_h100_preflight_v2/independent_audit.json').read_text())
@app.local_entrypoint()
def preflight():print(json.dumps(component.remote()))
