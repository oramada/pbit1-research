"""Observe the existing production process, then run its full independent audit."""
import json,time,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
from .positional_operator_common import OUT,sha,guard
from .gpu_environment import cuda_environment
from .block_cached_control_worker import atomic_json
PID=3633671

def identity():
    path=Path(f'/proc/{PID}')
    try:
        state=path.joinpath('stat').read_text().rsplit(')',1)[1].split();cmd=path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    except FileNotFoundError:return None
    return {'start_ticks':state[19],'state':state[0],'command':cmd}

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);initial=identity();assert initial is not None and 'research.thrml_spw.positional_operators' in initial['command'] and initial['state']!='Z';sources={str(Path(__file__)):sha(__file__),'research/thrml_spw/launch_positional_operator_audit.py':sha('research/thrml_spw/launch_positional_operator_audit.py')};record={'phase':'waiting_on_actual_production_process','watcher_pid':os.getpid(),'producer_pid':PID,'producer_identity':initial,'source_sha256':sources,'protocol_sha256':sha(OUT/'protocol.json'),'recorded_utc':datetime.now(timezone.utc).isoformat()};checkpoint=OUT/'AUDIT_PIPELINE_CHECKPOINT.json';atomic_json(checkpoint,record)
    while True:
        now=identity()
        if now is None or now['state']=='Z':break
        assert now['start_ticks']==initial['start_ticks'] and now['command']==initial['command'],'Process identity changed; do not infer success.'
        record.update({'verified_live_utc':datetime.now(timezone.utc).isoformat(),'producer_state':now['state'],'completed_operator_outputs':len(list((OUT/'outputs').glob('*.json')))});atomic_json(checkpoint,record);time.sleep(30)
    assert (OUT/'manifest.json').exists() and (OUT/'statistics.json').exists(),'Original producer is terminal without completed manifest. Do not restart or sample.'
    guard(p)
    for source,h in sources.items():assert sha(source)==h
    environment=cuda_environment();environment['XLA_FLAGS']='--xla_gpu_autotune_level=0';record['phase']='independent_native_audit';atomic_json(checkpoint,record);log=(OUT/'independent_audit.log').open('w')
    try:
        child=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.launch_positional_operator_audit'],env=environment,stdout=log,stderr=subprocess.STDOUT);record['audit_child_pid']=child.pid;atomic_json(checkpoint,record);code=child.wait();record['audit_exit_code']=code;atomic_json(checkpoint,record);assert code==0,'Independent native audit failed; preserve original data and failed gates.'
    finally:log.close()
    assert json.loads((OUT/'evidence_audit.json').read_text())['passed'];record['phase']='report_generation';atomic_json(checkpoint,record)
    with (OUT/'report.log').open('w') as report_log:code=subprocess.call([str(Path('.venv-thrml/bin/python').absolute()),'-m','research.thrml_spw.positional_operator_report'],stdout=report_log,stderr=subprocess.STDOUT)
    record.update({'phase':'operator_audit_and_report_complete' if code==0 else 'report_failed','report_exit_code':code,'completed_utc':datetime.now(timezone.utc).isoformat(),'goal_status':'active','full_student_scoring_started':False});atomic_json(checkpoint,record);assert code==0;print(json.dumps(record,indent=2))
if __name__=='__main__':main()
