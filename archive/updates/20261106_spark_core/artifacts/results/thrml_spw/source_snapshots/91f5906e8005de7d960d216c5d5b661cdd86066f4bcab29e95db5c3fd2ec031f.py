"""Wait for full parent audit, then GPU proof/controls/full matched comparison."""
from pathlib import Path
import json,time,os,subprocess,datetime,shutil
from results.thrml_spw.fashion_cnn_native.word_comparison.common import OUT,PARENT,sha,write,guard,parent_closed
def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);path=OUT/'pipeline.json';assert not path.exists(),'No duplicate matched-word runner';OUT.joinpath('logs').mkdir(exist_ok=True);state={'status':'waiting_for_single_node_CNN_full_audits','stages':[]};write(path,state)
    while True:
        guard(p);parent=json.loads((PARENT/'pipeline.json').read_text())
        if parent['status']=='full_native_CNN_independently_audited_goal_active':parent_closed();break
        proc=Path('/proc')/str(p['parent_pid']);alive=proc.exists() and proc.joinpath('stat').read_text().split()[21]==p['parent_start_ticks']
        if parent['status']=='failed' or not alive:
            again=json.loads((PARENT/'pipeline.json').read_text())
            if again['status']=='full_native_CNN_independently_audited_goal_active':parent_closed();break
            state.update(status='parent_dependency_terminal_before_success',parent_status=again['status'],parent_live=alive);write(path,state);raise SystemExit(1)
        state['observed_parent_live_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(path,state);time.sleep(30)
    assert shutil.disk_usage('.').free>=p['minimum_free_before_stages'];gpu=['.venv/bin/python','-m','research.thrml_spw.gpu_environment','--','.venv-thrml-gpu/bin/python','-m'];cpu=['.venv-thrml/bin/python','-m'];torch=['.venv/bin/python','-m'];prefix='results.thrml_spw.fashion_cnn_native.word_comparison.'
    for name,runner,module,args in [('GPU_validation',gpu,'preflight',[]),('nominal_control',cpu,'controls',[]),('production',gpu,'study',['run']),('public_native_audit',gpu,'audit',[]),('statistics',cpu,'statistics',[]),('independent_statistics_report',torch,'verify',['run'])]:
        guard(p);command=runner+[prefix+module]+args;row={'stage':name,'command':command,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};state['status']='running';state['stages'].append(row);write(path,state);log=OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','XLA_FLAGS':'--xla_gpu_autotune_level=0'});row['pid']=child.pid;write(path,state);code=child.wait()
        row.update(returncode=code,status='completed' if code==0 else 'failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(log));write(path,state)
        if code:state['status']='failed';write(path,state);raise SystemExit(code)
    state['status']='full_matched_word_CNN_independently_audited_goal_active';write(path,state)
if __name__=='__main__':main()
