"""Source-seal conventional controls and wait on the actual full CNN runner."""
import json,datetime,time,subprocess,os,shutil
from importlib.metadata import version
from pathlib import Path
from results.thrml_spw.cifar_cnn_native.precision_baselines.common import ROOT,OUT,TRAIN,PARENT,NAMES,sha,write,guard,parent_closed,parent_controls
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists();check=json.loads((ROOT/'precision_selfcheck.json').read_text());assert check['passed']
    sources={str(f):sha(f) for f in sorted((ROOT/'precision_baselines').glob('*.py'))};sources.update(json.loads((PARENT/'protocol.json').read_text())['source_sha256']);sources[str(ROOT/'model.py')]=sha(ROOT/'model.py')
    for f,h in check['source_sha256'].items():assert sha(f)==h,f
    train=json.loads((TRAIN/'manifest.json').read_text());files=[TRAIN/'manifest.json',TRAIN/'validation.npz',ROOT/'precision_selfcheck.json',PARENT/'protocol.json']+[Path(r[k]) for r in train['rows'] for k in ['path','checkpoint_path']];stat=Path('/proc/3964689/stat').read_text().split();assert stat[21]=='47733621'
    p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'input_sha256':{str(f):sha(f) for f in files},'parent_protocol_sha256':sha(PARENT/'protocol.json'),'parent_pid':3964689,'parent_start_ticks':stat[21],'conditions':NAMES,'models':5,'images':10000,'samples':[64,256,1024],'batch':128,'new_native_updates':0,'environment':{'torch':version('torch'),'numpy':version('numpy'),'CPU_threads':4,'deterministic_algorithms':True,'GPU_used':False},'extra_storage_guard_bytes':50000000,'scope':'Four conventional FP16/BF16 controls, fixed before test scoring. Weight-storage-only F64 arithmetic and actual pinned TorchCPU low-precision outputs distinguished; internal accumulator precision unclaimed. All12paired contrasts reuse parent hierarchicalbootstrap draws on historically consumed data, not fresh independent MC/generalization. No physical hardware energy/latency/area or outcome-selected gate.'};write(OUT/'protocol.json',p);state={'status':'waiting_for_parent_controls','stages':[]};write(OUT/'pipeline.json',state)
    def wait_parent(require_complete):
        while True:
            guard(p);parent=json.loads((PARENT/'pipeline.json').read_text())
            if parent['status']=='all150compactRGBnative_cases_independently_audited_goal_active':parent_closed(p);return
            if not require_complete and (PARENT/'controls.json').exists():parent_controls(p);return
            proc=Path('/proc')/str(p['parent_pid']);alive=proc.exists() and proc.joinpath('stat').read_text().split()[21]==p['parent_start_ticks']
            if parent['status']=='failed' or not alive:
                again=json.loads((PARENT/'pipeline.json').read_text())
                if again['status']=='all150compactRGBnative_cases_independently_audited_goal_active':parent_closed(p);return
                state.update(status='parent_dependency_terminal_before_success',parent_status=again['status'],parent_live=alive);write(OUT/'pipeline.json',state);raise SystemExit(1)
            state['observed_parent_live_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(OUT/'pipeline.json',state);time.sleep(30)
    wait_parent(False);assert shutil.disk_usage('.').free>=p['extra_storage_guard_bytes']
    for stage,module,args in [('evaluate','run',['evaluate']),('statistics','run',['statistics']),('independent_audit_report','verify',['run'])]:
        if stage=='statistics':state['status']='waiting_for_parent_full_audits';write(OUT/'pipeline.json',state);wait_parent(True)
        guard(p);command=['.venv/bin/python','-m','results.thrml_spw.cifar_cnn_native.precision_baselines.'+module]+args;row={'stage':stage,'command':command,'status':'running'};state['stages'].append(row);state['status']='running';write(OUT/'pipeline.json',state)
        with (OUT/(stage+'.log')).open('wb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'});row['pid']=child.pid;write(OUT/'pipeline.json',state);code=child.wait()
        row.update(returncode=code,status='completed' if code==0 else 'failed');write(OUT/'pipeline.json',state)
        if code:state['status']='failed';write(OUT/'pipeline.json',state);raise SystemExit(code)
    state['status']='FP16_BF16_RGB_controls_audited_goal_active';write(OUT/'pipeline.json',state)
if __name__=='__main__':main()
