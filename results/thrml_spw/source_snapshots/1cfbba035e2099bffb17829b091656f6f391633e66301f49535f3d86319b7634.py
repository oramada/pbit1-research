"""Actual GPT2 uint32 count installation and replay on synthetic input only."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-highT-model-preflight-v1')
PROTOCOL='results/thrml_spw/broad_verification_20261105/highT_preparation_v1/model_preflight/protocol.json'

@app.function(image=image,gpu='H100!',cpu=8,memory=65536,volumes={'/workspace':volume},
              timeout=7200,retries=0,max_containers=1,scaledown_window=2)
def preflight(protocol_sha256:str):
    import subprocess,os,json,hashlib
    setup_workspace();volume.reload();path=Path('/workspace')/PROTOCOL
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol_sha256
    log=Path('/workspace/highT_model_preflight_v1.log')
    with log.open('x')as f:
        for module in ['highT_case_v1','verify_highT_case_v1']:
            r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.broad.'+module,
                '--protocol',PROTOCOL,'--case','gpt2__nativeT1'],cwd='/workspace',
                env={**os.environ,'PYTHONPATH':'/code'},stdout=f,stderr=f)
            f.flush();volume.commit();assert r.returncode==0,{'exit_code':r.returncode,'module':module,'log':str(log)}
    p=Path('/workspace/results/thrml_spw/broad_verification_20261105/highT_preparation_v1/model_preflight/cases/gpt2__nativeT1/independent_audit.json')
    return json.loads(p.read_text())
