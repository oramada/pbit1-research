"""Durable H100 R1 cases; all outcomes retained, no automatic sampling retry."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
from cloud.highT_execution_v1.layout import APP_NAME,PROTOCOL,OUTPUT,LOG_DIRECTORY,CASE_MODULES,STATISTICS_MODULES
app=modal.App(APP_NAME)
claims=modal.Dict.from_name(APP_NAME,create_if_missing=True)


def contract(protocol_sha256):
    import hashlib,json
    setup_workspace();volume.reload();path=Path('/workspace')/PROTOCOL
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol_sha256
    p=json.loads(path.read_text());assert p['status']=='complete_R1_execution_and_statistics_frozen_before_scoring'
    return p


@app.function(image=image,gpu='H100!',cpu=8,memory=65536,volumes={'/workspace':volume},
              timeout=86400,retries=0,max_containers=4,scaledown_window=5)
def case(slug:str,protocol_sha256:str):
    import os,json,time,subprocess,uuid,traceback
    p=contract(protocol_sha256);assert slug in p['cases'];call=modal.current_function_call_id();key=protocol_sha256+':'+slug
    owner={'call_id':call,'case':slug};won=claims.put(key,owner,skip_if_exists=True);previous=claims.get(key)
    if not won and previous['call_id']!=call:return {'status':'duplicate_rejected_reattach_to_owner','owner':previous}
    root=(Path('/workspace')/LOG_DIRECTORY/slug);root.mkdir(parents=True,exist_ok=True)
    if list(root.glob('terminal_failure_*.json')):raise RuntimeError('Preserved program failure is terminal')
    attempt=uuid.uuid4().hex;logpath=root/(attempt+'.log')
    try:
        with logpath.open('x') as log:
            for module in CASE_MODULES:
                process=subprocess.Popen(['/opt/torch/bin/python','-m','research.thrml_spw.broad.'+module,
                    '--protocol',PROTOCOL,'--case',slug],cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code',
                    'CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=log)
                while process.poll()is None:time.sleep(15);log.flush();volume.commit()
                log.flush();volume.commit();assert process.returncode==0,{'case':slug,'module':module,'exit_code':process.returncode,'log':str(logpath)}
        audit=json.loads((Path('/workspace')/OUTPUT/'cases'/slug/'independent_audit.json').read_text())
        assert audit['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
        assert not audit['synthetic_only'] and audit['protocol_sha256']==protocol_sha256
    except Exception as e:
        (root/('terminal_failure_'+attempt+'.json')).write_text(json.dumps({'case':slug,'type':type(e).__name__,
            'message':str(e),'traceback':traceback.format_exc(),'call_id':call},indent=2)+'\n');volume.commit();raise
    result={'status':'R1_case_and_independent_neural_replay_complete','owner':owner}
    claims.put(key+':result',result);return result


@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=7200,
              retries=0,max_containers=1,scaledown_window=2,nonpreemptible=True)
def statistics(protocol_sha256:str):
    import os,subprocess,json,uuid
    p=contract(protocol_sha256)
    call=modal.current_function_call_id();key=protocol_sha256+':statistics';owner={'call_id':call}
    won=claims.put(key,owner,skip_if_exists=True);previous=claims.get(key)
    if not won and previous['call_id']!=call:return {'status':'duplicate_rejected_reattach_to_owner','owner':previous}
    logpath=Path('/workspace')/LOG_DIRECTORY/('statistics_'+uuid.uuid4().hex+'.log')
    logpath.parent.mkdir(exist_ok=True)
    with logpath.open('x')as log:
        for module in STATISTICS_MODULES:
            filename='followup_audit.json'if module.startswith('verify_')else'followup.json'
            destination=Path('/workspace')/OUTPUT/'statistics'/filename
            if destination.exists():
                existing=json.loads(destination.read_text());assert existing['protocol_sha256']==protocol_sha256;continue
            r=subprocess.run(['/opt/torch/bin/python','-m','research.thrml_spw.broad.'+module,'--protocol',PROTOCOL],
                cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code'},stdout=log,stderr=log)
            log.flush();volume.commit();assert r.returncode==0,{'module':module,'code':r.returncode,'log':str(logpath)}
    result=json.loads((Path('/workspace')/OUTPUT/'statistics/followup_audit.json').read_text())
    assert result['status']=='highT_followup_full_fixed_cohort_statistics_and_all12case_bindings_independently_verified'
    return result
