"""Continue the authorized research only after the actual operator audit exits."""
import os,json,time,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone
from .positional_student_common import SOURCES,sha
from .fine_range_common import atomic_json
ROOT=Path('results/thrml_spw/positional_operators')
OUT=Path('results/thrml_spw/positional_student')
UPSTREAM_PID=3647421

def identity():
    folder=Path(f'/proc/{UPSTREAM_PID}')
    try:
        stat=folder.joinpath('stat').read_text().rsplit(')',1)[1].split();cmd=folder.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    except FileNotFoundError:return None
    return {'start_ticks':stat[19],'state':stat[0],'command':cmd}

def vendor():
    commit=subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip();assert commit=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';assert not subprocess.check_output(['git','-C','vendor/thrml','status','--porcelain'],text=True).strip();return commit

def main():
    initial=identity();assert initial and initial['state']!='Z' and 'research.thrml_spw.wait_positional_operator_audit' in initial['command'];assert not (OUT/'protocol.json').exists();vendor();sources={str(Path('research/thrml_spw')/name):sha(Path('research/thrml_spw')/name) for name in SOURCES+['verify_positional_operator_summary.py']};sources[str(Path(__file__))]=sha(__file__)
    for path,h in sources.items():
        archive=Path('results/thrml_spw/source_snapshots')/(h+'.py');data=Path(path).read_bytes()
        if archive.exists():assert archive.read_bytes()==data
        else:archive.write_bytes(data)
    record={'phase':'waiting_for_actual_operator_audit_process','watcher_pid':os.getpid(),'upstream_pid':UPSTREAM_PID,'upstream_identity':initial,'source_sha256':sources,'prospective_neural_plan_sha256':sha(ROOT/'NEXT_NEURAL_STUDY.md'),'queued_utc':datetime.now(timezone.utc).isoformat(),'source_code_sealed_at_queue':True,'full_student_protocol_prepared':False,'new_student_scoring_started':False,'goal_status':'active','no_original_sampling_restart':True};checkpoint=ROOT/'STUDENT_PIPELINE_CHECKPOINT.json';atomic_json(checkpoint,record)
    def guards():
        for path,h in sources.items():assert sha(path)==h,path
        assert sha(ROOT/'NEXT_NEURAL_STUDY.md')==record['prospective_neural_plan_sha256'];vendor()
    def stage(name,interpreter,module,extra=None):
        guards();record['phase']=name;record['phase_started_utc']=datetime.now(timezone.utc).isoformat();atomic_json(checkpoint,record);logs=ROOT/'student_pipeline_logs';logs.mkdir(exist_ok=True)
        with (logs/f'{name}.log').open('w') as log:
            command=[str(Path(interpreter).absolute()),'-u']
            command+=(['-m','unittest',module] if extra=='unittest' else ['-m',module])
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4'});record['current_child_pid']=process.pid;atomic_json(checkpoint,record);code=process.wait()
        record.setdefault('terminal_stages',[]).append({'stage':name,'exit_code':code,'log':str(logs/f'{name}.log'),'log_sha256':sha(logs/f'{name}.log')});atomic_json(checkpoint,record);assert code==0,f'{name} failed; preserve original run/log, do not restart sampling.'
    try:
        while True:
            now=identity()
            if now is None or now['state']=='Z':break
            assert now['start_ticks']==initial['start_ticks'] and now['command']==initial['command'],'Upstream PID identity changed.';record['last_verified_upstream_live_utc']=datetime.now(timezone.utc).isoformat();atomic_json(checkpoint,record);time.sleep(30)
        upstream=json.loads((ROOT/'AUDIT_PIPELINE_CHECKPOINT.json').read_text());assert upstream['phase']=='operator_audit_and_report_complete' and upstream['audit_exit_code']==upstream['report_exit_code']==0
        for name in ['control_audit.json','evidence_audit.json']:assert json.loads((ROOT/name).read_text())['passed']
        stage('operator_summary_audit','.venv/bin/python','research.thrml_spw.verify_positional_operator_summary')
        stage('student_preflight_tests','.venv/bin/python','research.thrml_spw.test_positional_student_preflight','unittest')
        stage('student_protocol_preparation','.venv/bin/python','research.thrml_spw.prepare_positional_student');record['full_student_protocol_prepared']=True;record['student_protocol_sha256']=sha(OUT/'protocol.json');atomic_json(checkpoint,record)
        record['new_student_scoring_started']=True;atomic_json(checkpoint,record);stage('student_production','.venv/bin/python','research.thrml_spw.positional_student')
        stage('student_original_model_and_public_native_audit','.venv/bin/python','research.thrml_spw.audit_positional_student')
        stage('student_paired_statistics','.venv/bin/python','research.thrml_spw.positional_student_statistics')
        stage('student_independent_statistics_audit','.venv/bin/python','research.thrml_spw.verify_positional_student_statistics')
        stage('student_report','.venv/bin/python','research.thrml_spw.positional_student_report')
        completion={'status':'completed_and_independently_audited_goal_active','recorded_utc':datetime.now(timezone.utc).isoformat(),'goal_status':'active','neural_cases':10,'native_projection_bank_calls':960,'production_node_updates':434865438720,'independent_native_replay_updates':869730877440,'new_heldout_test_scores':False,'physical_systems_advantage_proven':False,'original_prior_quality_gates_unchanged':True,'source_pipeline_sha256':sha(__file__),'artifact_sha256':{name:sha(OUT/name) for name in ['protocol.json','manifest.json','model_audit.json','native_audit.json','statistics.json','statistics_audit.json','resource_ledger.json']}};atomic_json(OUT/'completion.json',completion);record.update({'phase':'causal_student_study_audited_report_complete','completed_utc':datetime.now(timezone.utc).isoformat(),'current_child_pid':None,'goal_status':'active','global_snapshot_pending_root_review':True});atomic_json(checkpoint,record);print(json.dumps(record,indent=2))
    except Exception as exc:
        record.update({'phase':'pipeline_failed_preserve_evidence','error':str(exc),'failed_utc':datetime.now(timezone.utc).isoformat(),'goal_status':'active'});atomic_json(checkpoint,record);traceback.print_exc();raise
if __name__=='__main__':main()
