"""Prepare once and wait on the actual integer-cohort parent before GPU work."""
from pathlib import Path
import json,hashlib,datetime,subprocess,os,time,shutil
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'throughput_evidence';PARENT=Path('results/thrml_spw/integer_activation_native/cohort')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,d):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(path)
def parent_closed():
    assert json.loads((PARENT/'pipeline.json').read_text())['status']=='full_integer_activation_cohort_audited_goal_active'
    for f,h in json.loads((PARENT/'completion.json').read_text())['artifact_sha256'].items():assert sha(f)==h,f
    assert json.loads((PARENT/'evidence_audit.json').read_text())['passed'] and json.loads((PARENT/'statistics_audit.json').read_text())['passed']
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'No duplicate GPU preflight watcher'
    assert json.loads((ROOT/'compact_witness/evidence_audit.json').read_text())['passed'] and json.loads((ROOT/'fast_witness/evidence_audit.json').read_text())['passed']
    sources=json.loads((ROOT/'fast_witness/protocol.json').read_text())['source_sha256'];sources.update({str(f):sha(f) for f in [*sorted((ROOT/'throughput').glob('*.py')),*sorted((ROOT/'compact').glob('*.py'))]})
    train=json.loads((ROOT/'training/manifest.json').read_text());inputs=[ROOT/'training/manifest.json',ROOT/'training/evidence_audit.json',ROOT/'training/validation.npz',Path(train['rows'][0]['path']),ROOT/'compact_witness/evidence_audit.json',ROOT/'fast_witness/evidence_audit.json'];parent_pid=3778471;stat=Path(f'/proc/{parent_pid}/stat').read_text().split();assert stat[21]=='44679072'
    p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'input_artifact_sha256':{str(f):sha(f) for f in inputs},'samples':[64,256,1024],'key_root':1527520,'model':0,'validation_inputs':16,'banks':2,'images_per_bank':8,'native_runs_per_budget':2,'public_recovery_runs_per_budget':1,'parent_pid':parent_pid,'parent_start_ticks':stat[21],'minimum_free_bytes':50000000,'scope':'Validation-only fixed full-budget GPU throughput/replay prerequisite for future CNN full test study. No labels, scores, outcome-selected budget, physical hardware timing or measured energy. Two native passes reuse exactly the same trajectories; public recovery is verification, not independent replication.'};write(OUT/'protocol.json',p);state={'status':'waiting_for_integer_parent_completion','stages':[]};write(OUT/'pipeline.json',state)
    while True:
        for file,h in p['source_sha256'].items():assert sha(file)==h,file
        parent=json.loads((PARENT/'pipeline.json').read_text())
        if parent['status']=='full_integer_activation_cohort_audited_goal_active':parent_closed();break
        proc=Path('/proc')/str(parent_pid);alive=proc.exists() and proc.joinpath('stat').read_text().split()[21]==p['parent_start_ticks']
        if parent['status']=='failed' or not alive:
            again=json.loads((PARENT/'pipeline.json').read_text())
            if again['status']=='full_integer_activation_cohort_audited_goal_active':parent_closed();break
            state.update(status='parent_dependency_terminal_before_success',parent_status=again['status'],parent_live=alive);write(OUT/'pipeline.json',state);raise SystemExit(1)
        state['observed_parent_live_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(OUT/'pipeline.json',state);time.sleep(30)
    assert shutil.disk_usage('.').free>=p['minimum_free_bytes']
    command=['.venv/bin/python','-m','research.thrml_spw.gpu_environment','--','.venv-thrml-gpu/bin/python','-m','results.thrml_spw.fashion_cnn_native.throughput.run'];row={'stage':'GPU_full_budget_validation_and_public_recovery','command':command,'status':'running'};state.update(status='running');state['stages'].append(row);write(OUT/'pipeline.json',state)
    with (OUT/'execution.log').open('wb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4','XLA_FLAGS':'--xla_gpu_autotune_level=0'});row['pid']=child.pid;write(OUT/'pipeline.json',state);code=child.wait()
    row.update(returncode=code,status='completed' if code==0 else 'failed');state['status']='GPU_CNN_validation_preflight_audited_goal_active' if code==0 else 'failed';write(OUT/'pipeline.json',state);raise SystemExit(code)
if __name__=='__main__':main()
