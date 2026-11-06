"""H100fulltile primary-budget fixture, separate from active neuralcase."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-additional-large-tile-v1')

@app.function(image=image,gpu='H100!',cpu=4,memory=16384,volumes={'/workspace':volume},
              timeout=1800,retries=0,max_containers=1,scaledown_window=2)
def preflight():
    import os,subprocess,json
    setup_workspace();volume.reload()
    path=Path('/workspace/additional_large_tile_v1.log')
    with path.open('x') as log:
        for module in ['preflight_large_tile_h100_v1','verify_large_tile_h100_v1']:
            cmd=['/opt/torch/bin/python','-m','research.thrml_spw.gpu_environment','--',
                '/opt/thrml/bin/python','-m','research.thrml_spw.broad.'+module]
            r=subprocess.run(cmd,cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            log.flush();volume.commit();assert r.returncode==0,{'module':module,'returncode':r.returncode,'log':str(path)}
    return json.loads(Path('/workspace/results/thrml_spw/broad_verification_20261105/large_tile_h100_v1/independent_audit.json').read_text())
