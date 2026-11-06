"""Bounded H100 raw-state/count fixture for additional-model R1 programs."""
import json
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-additional-counts-preflight-v1')

@app.function(image=image,gpu='H100!',cpu=4,memory=8192,volumes={'/workspace':volume},
              timeout=900,retries=0,max_containers=1,scaledown_window=2)
def preflight():
    import subprocess,os,uuid
    setup_workspace();volume.reload()
    path=Path('/workspace/additional_counts_h100_v1_'+uuid.uuid4().hex+'.log')
    with path.open('x') as log:
        for module in ['preflight_parameter_counts_h100_v1','verify_parameter_counts_h100_v1']:
            cmd=['/opt/torch/bin/python','-m','research.thrml_spw.gpu_environment','--',
                '/opt/thrml/bin/python','-m','research.thrml_spw.broad.'+module]
            r=subprocess.run(cmd,cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            log.flush();volume.commit();assert r.returncode==0,{'module':module,'returncode':r.returncode,'log':str(path)}
    return json.loads(Path('/workspace/results/thrml_spw/broad_verification_20261105/parameter_counts_h100_v1/independent_audit.json').read_text())
