"""CPU-only read-only diagnosis of already completed primary countfiles."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-primary-count-moments-v1')
@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=3600,
              retries=0,max_containers=1,scaledown_window=2,nonpreemptible=True)
def diagnose():
    import os,json,subprocess
    setup_workspace();volume.reload();path=Path('/workspace/primary_count_moments_v1.log')
    with path.open('x') as log:
        r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.broad.diagnose_primary_counts_v1'],
            cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
        log.flush();volume.commit();assert r.returncode==0,{'returncode':r.returncode,'log':str(path)}
    return json.loads(Path('/workspace/results/thrml_spw/broad_verification_20261105/primary_count_moments_v1/results.json').read_text())
