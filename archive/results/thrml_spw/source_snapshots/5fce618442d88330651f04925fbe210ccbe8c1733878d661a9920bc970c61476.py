"""Sequential run/audits/statistics with terminal receipts, never timeout restarts."""
import json,os,subprocess,sys,datetime
from pathlib import Path
from .boolean_classifier_common import OUT,sha,guard
from .fine_range_common import atomic_json

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);path=OUT/'pipeline.json'
    state={'stages':[],'status':'running','pipeline_source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json')}
    assert not path.exists(),'No automatic duplicate pipeline'
    OUT.joinpath('logs').mkdir(exist_ok=True);atomic_json(path,state)
    gpu=[str(Path('.venv/bin/python')),'-m','research.thrml_spw.gpu_environment','--',str(Path('.venv-thrml-gpu/bin/python')),'-m']
    cpu=[str(Path('.venv-thrml/bin/python')),'-m']
    stages=[('production',gpu,'boolean_classifier_study',['run']),('native_audit',gpu,'verify_boolean_classifier_study',[]),('statistics',cpu,'boolean_classifier_statistics',[]),('statistics_audit',cpu,'verify_boolean_classifier_statistics',[]),('report',cpu,'finalize_boolean_classifier',[])]
    for name,prefix,module,args in stages:
        guard(p);command=prefix+['research.thrml_spw.'+module]+args;log=OUT/'logs'/(name+'.log');row={'stage':name,'command':command,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'log':str(log),'status':'running'};state['stages'].append(row);atomic_json(path,state)
        env={**os.environ,'OPENBLAS_NUM_THREADS':'4','XLA_FLAGS':'--xla_gpu_autotune_level=0'}
        with log.open('wb') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env);row['pid']=child.pid;atomic_json(path,state);returncode=child.wait()
        row.update(returncode=returncode,status='completed' if returncode==0 else 'failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(log));atomic_json(path,state)
        if returncode:
            state['status']='failed';atomic_json(path,state);raise SystemExit(returncode)
    state['status']='completed_and_independently_audited_goal_active';atomic_json(path,state)
if __name__=='__main__':main()
