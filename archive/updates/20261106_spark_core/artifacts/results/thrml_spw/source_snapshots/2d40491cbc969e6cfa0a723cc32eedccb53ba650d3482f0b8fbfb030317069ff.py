"""Verification-only replay of one already-closed nonzero LFM3 batch."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-lfm3-shard-proof-v1')
TRANSPORT='results/thrml_spw/broad_verification_20261105/lfm3_sharding_preparation_v1/proof_transport.json'
CORE='results/thrml_spw/lfm3_1p7b/holdout_native_v4_protocol.json'

@app.function(image=image,gpu='H100!',cpu=8,memory=65536,volumes={'/workspace':volume},
              timeout=7200,retries=0,max_containers=1,scaledown_window=2)
def replay(transport_sha256:str):
    import os,subprocess,hashlib,json,uuid
    setup_workspace();volume.reload();path=Path('/workspace')/TRANSPORT
    assert hashlib.sha256(path.read_bytes()).hexdigest()==transport_sha256
    logs=Path('/workspace/cloud_lfm3_shard_proof_v1');logs.mkdir(exist_ok=True)
    log=logs/(uuid.uuid4().hex+'.log')
    with log.open('x')as f:
        commands=[['holdout_native_case_shard_v1','--seed','202611150','--protocol',CORE,'--transport',TRANSPORT,'--shard','proof4'],
                  ['verify_holdout_shard_replay_v1','--transport',TRANSPORT]]
        for args in commands:
            proc=subprocess.Popen(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.'+args[0],*args[1:]],cwd='/workspace',
                env={**os.environ,'PYTHONPATH':'/code'},stdout=f,stderr=f)
            while proc.poll()is None:__import__('time').sleep(15);f.flush();volume.commit()
            f.flush();volume.commit();assert proc.returncode==0,{'module':args[0],'code':proc.returncode,'log':str(log)}
    result=Path('/workspace/results/thrml_spw/lfm3_1p7b/holdout_sharding_proof_v1/202611150/proof4/replay_audit.json')
    return json.loads(result.read_text())
