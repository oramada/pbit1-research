"""Bounded production-width count-only H100 timing."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-highT-tile-benchmark-v1')

@app.function(image=image,gpu='H100!',cpu=4,memory=16384,volumes={'/workspace':volume},
              timeout=3600,retries=0,max_containers=1,scaledown_window=2)
def benchmark():
    import subprocess,os,json
    setup_workspace();volume.reload();log=Path('/workspace/highT_tile_benchmark_v1.log')
    with log.open('x')as f:
        r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.gpu_environment','--',
            '.venv-thrml-gpu/bin/python','-m','research.thrml_spw.broad.highT_tile_benchmark_v1'],
            cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=f,stderr=f)
        f.flush();volume.commit();assert r.returncode==0,{'exit_code':r.returncode,'log':str(log)}
    return json.loads(Path('/workspace/results/thrml_spw/broad_verification_20261105/highT_preparation_v1/tile_benchmark_gpu/results.json').read_text())
