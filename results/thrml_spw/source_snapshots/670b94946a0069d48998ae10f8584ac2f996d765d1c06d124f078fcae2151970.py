"""Wait on the original parent handle, then run the sealed derived control once."""
from pathlib import Path
import json,hashlib,datetime,time,subprocess,sys
OUT=Path('results/thrml_spw/boolean_classifier_cached_reuse');PARENT=Path('results/thrml_spw/boolean_classifier')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
    e=json.loads((OUT/'execution_protocol.json').read_text());path=OUT/'continuation.json';assert not path.exists();state={'status':'waiting_for_original_parent','parent_pid':e['parent_pipeline_pid'],'parent_start_ticks':e['parent_pipeline_start_ticks'],'stages':[]};write(path,state)
    while True:
        for f,h in e['source_sha256'].items():assert sha(f)==h
        p=json.loads((PARENT/'pipeline.json').read_text())
        if p['status']=='completed_and_independently_audited_goal_active':break
        proc=Path('/proc')/str(e['parent_pipeline_pid']);alive=proc.exists() and proc.joinpath('stat').read_text().split()[21]==e['parent_pipeline_start_ticks']
        if p['status']=='failed' or not alive:
            state.update(status='parent_dependency_terminal_before_success',parent_status=p['status'],parent_live=alive);write(path,state);raise SystemExit(1)
        state['observed_live_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(path,state);time.sleep(30)
    for stage,file in [('evaluation','evaluate.py'),('independent_audit_report','verify.py')]:
        command=[sys.executable,str(OUT/file)];row={'stage':stage,'command':command,'status':'running'};state.update(status='running_derived_control');state['stages'].append(row);write(path,state)
        with (OUT/(stage+'.log')).open('wb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);row['pid']=child.pid;write(path,state);code=child.wait()
        row.update(returncode=code,status='completed' if code==0 else 'failed');write(path,state)
        if code:state['status']='failed';write(path,state);raise SystemExit(code)
    state['status']='derived_control_audited_goal_active';write(path,state)
if __name__=='__main__':main()
