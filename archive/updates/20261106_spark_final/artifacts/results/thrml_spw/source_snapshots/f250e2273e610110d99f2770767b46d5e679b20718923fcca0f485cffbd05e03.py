"""CPU-only pinned additional-model asset preparation, no neural scoring."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/code')
app=modal.App('pbit1-additional-assets-v1')
volume=modal.Volume.from_name('spw-h100-validation-v1')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy==2.5.3','huggingface_hub==2.3.0','transformers==5.21.0','tokenizers==0.23.3','pyarrow==26.0.0')
    .env({'PYTHONPATH':'/code','TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(ROOT/'research','/code/research',copy=True,ignore=['**/__pycache__/**']))

@app.function(image=image,cpu=4,memory=8192,volumes={'/workspace':volume},timeout=3600,
              retries=0,max_containers=1,scaledown_window=2,nonpreemptible=True)
def prepare():
    import os,sys,subprocess,json
    volume.reload()
    p=Path('/workspace/additional_assets_v1.log')
    with p.open('x') as log:
        r=subprocess.run([sys.executable,'-m','research.thrml_spw.broad.prepare_assets_v1'],cwd='/workspace',
            env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
        log.flush();volume.commit();assert r.returncode==0,{'returncode':r.returncode,'log':str(p)}
    return json.loads(Path('/workspace/results/thrml_spw/broad_verification_20261105/additional_assets_v1/manifest.json').read_text())
