"""H100 execution; explicit user authorization received before deployment."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2]
app = modal.App('spw-h100-validation-v2')
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
              ephemeral_disk=131072, volumes={'/workspace': volume},
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
              ephemeral_disk=131072, volumes={'/workspace': volume},
              timeout=14400, retries=0, max_containers=4)
def run_case(module: str, arguments: list[str], label: str):
    """Only versioned cloud runners may write new case namespaces."""
    import os
    import subprocess
    assert module.startswith('cloud.modal_h100_v1.')
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
