"""Separate CPU raw-checkpoint oracle, no producer reexecution or quality."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-additional-packages-audit-v1')

@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=3600,
              retries=0,max_containers=1,scaledown_window=2,nonpreemptible=True)
def audit():
    import os,subprocess,json
    setup_workspace();volume.reload()
    path=Path('/workspace/additional_parameter_packages_independent_audit_v1.log')
    with path.open('x') as log:
        r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.broad.verify_parameter_packages_v1'],
            cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
        log.flush();volume.commit();assert r.returncode==0,{'returncode':r.returncode,'log':str(path)}
    return json.loads(Path('/workspace/results/thrml_spw/broad_verification_20261105/additional_parameter_packages_v1/independent_audit.json').read_text())
