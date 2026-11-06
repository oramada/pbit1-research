"""Durable deployed H100 heldout jobs, atomic ownership and stage checkpoints."""
import json
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image, setup_workspace, volume

app = modal.App('pbit1-broad-heldout-v3')
claims = modal.Dict.from_name('pbit1-broad-heldout-v3', create_if_missing=True)


@app.function(image=image,cpu=2,memory=2048,volumes={'/workspace':volume},timeout=900,
              retries=0,max_containers=1,scaledown_window=2)
def legacy_inventory():
    import hashlib
    setup_workspace();volume.reload()
    root=Path('/workspace/results/thrml_spw/lfm3_1p7b/holdout_native_v2/202611150')
    result={}
    for p in sorted(root.glob('batch*.npz')):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
        result[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    assert set(result)=={'batch0_BF16.npz','batch0_same_fit_BF16.npz','batch0_nominal_controls.npz',
                         'batch0_native.npz','batch1_BF16.npz','batch1_same_fit_BF16.npz'}
    return {'status':'preserved_legacy_outputs_hashed_without_quality_inspection','seed':202611150,'files':result}


@app.function(image=image,gpu='H100!',cpu=8,memory=65536,ephemeral_disk=524288,
              volumes={'/workspace':volume},timeout=86400,retries=0,max_containers=4,scaledown_window=10)
def case(seed:int,protocol_sha256:str):
    import os,subprocess,time,uuid,hashlib,traceback
    call_id=modal.current_function_call_id()
    key=protocol_sha256+':'+str(seed)
    owner={'call_id':call_id,'protocol_sha256':protocol_sha256,'seed':seed}
    won=claims.put(key,owner,skip_if_exists=True)
    previous=claims.get(key)
    if not won and previous['call_id']!=call_id:
        return {'status':'duplicate_rejected_reattach_to_owner','owner':previous}
    setup_workspace();volume.reload()
    protocol=Path('/workspace/results/thrml_spw/lfm3_1p7b/holdout_native_v3_protocol.json')
    assert hashlib.sha256(protocol.read_bytes()).hexdigest()==protocol_sha256
    p=json.loads(protocol.read_text());assert seed in p['seeds']
    ready=json.loads(Path('/workspace/holdout_assets_v3.json').read_text())
    for name,record in ready['files'].items():
        f=Path('/workspace')/name
        assert f.stat().st_size==record['bytes']
        h=hashlib.sha256(f.read_bytes()).hexdigest()
        assert h==record['sha256'],name
    output=Path('/workspace/cloud_holdout_logs_v3')/str(seed)
    output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('terminal_failure_*.json')):
        raise RuntimeError('Preserved program failure is terminal; no automatic scientific retry')
    attempt=uuid.uuid4().hex
    logpath=output/(attempt+'.log')
    try:
        with logpath.open('x') as log:
            for module in ['holdout_native_case_v3','verify_holdout_native_case_v3']:
                process=subprocess.Popen(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.'+module,
                    '--seed',str(seed)],cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},
                    stdout=log,stderr=log)
                while process.poll() is None:
                    time.sleep(20)
                    log.flush();volume.commit()
                log.flush();volume.commit()
                assert process.returncode==0,{'seed':seed,'module':module,'exit_code':process.returncode,'log':str(logpath)}
    except Exception as error:
        (output/('terminal_failure_'+attempt+'.json')).write_text(json.dumps({'type':type(error).__name__,
            'message':str(error),'traceback':traceback.format_exc(),'call_id':call_id,'seed':seed},indent=2)+'\n')
        volume.commit();raise
    result={'seed':seed,'status':'heldout_case_and_independent_audit_complete','owner':owner}
    claims.put(key+':result',result);volume.commit();return result


@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=3600,
              retries=0,max_containers=1,scaledown_window=2,nonpreemptible=True)
def statistics(protocol_sha256:str):
    import os,subprocess,uuid,hashlib
    setup_workspace();volume.reload()
    protocol=Path('/workspace/results/thrml_spw/lfm3_1p7b/holdout_native_v3_protocol.json')
    assert hashlib.sha256(protocol.read_bytes()).hexdigest()==protocol_sha256
    path=Path('/workspace/cloud_holdout_logs_v3')/('statistics_'+uuid.uuid4().hex+'.log')
    with path.open('x') as log:
        for module in ['holdout_native_statistics_v3','verify_holdout_native_statistics_v3']:
            r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.lfm3.'+module],cwd='/workspace',
                env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            log.flush();volume.commit();assert r.returncode==0,{'module':module,'code':r.returncode,'log':str(path)}
    return json.loads(Path('/workspace/results/thrml_spw/lfm3_1p7b/holdout_native_v3/statistics_audit.json').read_text())
