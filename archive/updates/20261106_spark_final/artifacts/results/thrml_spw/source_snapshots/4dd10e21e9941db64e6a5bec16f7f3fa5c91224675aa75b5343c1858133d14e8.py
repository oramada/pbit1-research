"""Actual-model R1 replacement/replay on syntheticinputs only, twoH100jobs."""
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image,volume,setup_workspace
app=modal.App('pbit1-r1-model-preflight-v1')
claims=modal.Dict.from_name('pbit1-r1-model-preflight-v1',create_if_missing=True)

@app.function(image=image,gpu='H100!',cpu=4,memory=32768,volumes={'/workspace':volume},
              timeout=7200,retries=0,max_containers=2,scaledown_window=2)
def preflight(model:str,protocol_sha256:str):
    import os,subprocess,time,json,hashlib,uuid
    assert model in ['gpt2','distilbert'];call=modal.current_function_call_id();key=protocol_sha256+':'+model
    won=claims.put(key,{'call_id':call},skip_if_exists=True);owner=claims.get(key)
    if not won and owner['call_id']!=call:return {'status':'duplicate_rejected','owner':owner}
    setup_workspace();volume.reload()
    protocol='results/thrml_spw/broad_verification_20261105/r1_model_preflight_v1/protocol.json'
    assert hashlib.sha256((Path('/workspace')/protocol).read_bytes()).hexdigest()==protocol_sha256
    p=json.loads((Path('/workspace')/protocol).read_text());assert p['status']=='actual_model_synthetic_preflight_no_task_scoring'
    cases=[name for name,c in p['cases'].items() if c['model_key']==model];assert len(cases)==8
    logs=Path('/workspace/r1_model_preflight_logs_v1');logs.mkdir(exist_ok=True)
    done=[]
    for case in cases:
        logpath=logs/(case+'_'+uuid.uuid4().hex+'.log')
        with logpath.open('x') as log:
            for module in ['r1_case_v1','verify_r1_case_v1']:
                proc=subprocess.Popen(['/opt/torch/bin/python','-m','research.thrml_spw.broad.'+module,
                    '--protocol',protocol,'--case',case],cwd='/workspace',env={**os.environ,'PYTHONPATH':'/code',
                    'CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=log)
                while proc.poll()is None:time.sleep(10);log.flush();volume.commit()
                log.flush();volume.commit();assert proc.returncode==0,{'case':case,'module':module,'returncode':proc.returncode,'log':str(logpath)}
        audit=json.loads((Path('/workspace')/p['output']/'cases'/case/'independent_audit.json').read_text())
        assert audit['synthetic_only'] and audit['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
        done.append(audit)
    return {'status':'actual_model_all7controls_and_fullparameter_T1native_neural_replay_passed_synthetic_only',
        'model':model,'protocol_sha256':protocol_sha256,'cases':done,
        'scope':'Complete actualmodel parameterreplacement and output replay on synthetic IDs. No real evaluationtokens, taskquality or physicalhardwarebenefit.'}
