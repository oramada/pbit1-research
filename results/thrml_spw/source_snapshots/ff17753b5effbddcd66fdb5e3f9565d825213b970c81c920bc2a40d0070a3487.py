"""Wait on actual parent, then run native observer and all independent checks."""
from pathlib import Path
import json,subprocess,time,os,datetime
from results.thrml_spw.integer_activation_native.cohort_common import OUT,PARENT,ROOT,sha,guard,parent_closed,atomic_json

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);path=OUT/'pipeline.json';assert not path.exists(),'No implicit duplicate runner';(OUT/'logs').mkdir(exist_ok=True);state={'status':'waiting_for_parent_native_and_statistical_audits','source_sha256':sha(__file__),'stages':[]};atomic_json(path,state)
    while True:
        guard(p);parent=json.loads((PARENT/'pipeline.json').read_text())
        if parent['status']=='completed_and_independently_audited_goal_active':parent_closed();break
        proc=Path('/proc')/str(p['parent_pipeline_pid']);alive=proc.exists() and proc.joinpath('stat').read_text().split()[21]==p['parent_pipeline_start_ticks']
        if parent['status']=='failed' or not alive:
            # Re-read after disappearance to avoid a successful exit/status-read race.
            again=json.loads((PARENT/'pipeline.json').read_text())
            if again['status']=='completed_and_independently_audited_goal_active':parent_closed();break
            state.update(status='parent_dependency_terminal_before_success',parent_status=again['status'],parent_live=alive);atomic_json(path,state);raise SystemExit(1)
        state['observed_parent_live_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(path,state);time.sleep(30)
    gpu=['.venv/bin/python','-m','research.thrml_spw.gpu_environment','--','.venv-thrml-gpu/bin/python','-m'];cpu=['.venv-thrml/bin/python','-m'];prefix='results.thrml_spw.integer_activation_native.'
    for name,runner,module,args in [('production',gpu,'study',['run']),('paired_count_causal_audit',cpu,'audit',[]),('statistics',cpu,'statistics',[]),('independent_statistics_and_report',cpu,'verify_statistics',[])]:
        guard(p);command=runner+[prefix+module]+args;row={'stage':name,'command':command,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};state['status']='running';state['stages'].append(row);atomic_json(path,state);log=OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4','XLA_FLAGS':'--xla_gpu_autotune_level=0'});row['pid']=child.pid;atomic_json(path,state);code=child.wait()
        row.update(returncode=code,status='completed' if code==0 else 'failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(log));atomic_json(path,state)
        if code:state['status']='failed';atomic_json(path,state);raise SystemExit(code)
    state['status']='full_integer_activation_cohort_audited_goal_active';atomic_json(path,state)
if __name__=='__main__':main()
