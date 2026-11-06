"""Serial GPU verification of all closed GPT2 benchmark weight points."""
from pathlib import Path
import json,subprocess,sys,time,os
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'benchmark_replay_v1';root.mkdir(exist_ok=True);protocol=json.loads((OUT/'benchmark_v1/protocol.json').read_text());audit=json.loads((OUT/'benchmark_audit_v1/results.json').read_text())
    assert audit['status']=='all12cases_saved_array_coverage_and_statistics_audited'
    assert audit['benchmark_results_sha256']==sha(OUT/'benchmark_v1/results.json')
    results=[];start=time.monotonic();source=Path(__file__).with_name('replay_benchmark_v1.py')
    for c in protocol['cases']:
        name=c['name'];target=root/(name+'_neural.json');log=root/(name+'_neural.log')
        if target.exists():
            previous=json.loads(target.read_text());assert previous['source_sha256']==sha(source)and previous['neural_windows_replayed']==[0,293,585]
        else:
            with log.open('w')as f:
                p=subprocess.run([sys.executable,'-m','research.thrml_spw.spark_depth.replay_benchmark_v1','--case',name,'--with-forward'],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_DISABLE_PROGRESS_BARS':'1'})
            if p.returncode:
                (root/'replay_failure.json').write_text(json.dumps({'case':name,'returncode':p.returncode,'log':str(log)},indent=2)+'\n');raise RuntimeError('Neural replay failed: '+name)
        r=json.loads(target.read_text());assert r['status']=='passed_deterministic_point_replay'and r['neural_windows_replayed']==[0,293,585]
        results.append({'case':name,'receipt_sha256':sha(target),'point_sha256':r['point_sha256']})
        (root/'progress.json').write_text(json.dumps({'completed_cases':len(results),'total_cases':12,'last_case':name,'elapsed_seconds':time.monotonic()-start})+'\n');print(name,flush=True)
    (root/'all_results.json').write_text(json.dumps({'status':'all12_weight_points_and36_neural_windows_replayed','source_sha256':sha(__file__),'audit_sha256':sha(OUT/'benchmark_audit_v1/results.json'),'cases':results,'elapsed_seconds':time.monotonic()-start,
        'scope':'ExactGB10logitsandlossarrayson3prespecifiedwindowspercase; allweightpointsreconstructed. Samefrozenimplementation, notindependentneuralengine orfullsecondpassovereverywindow.'},indent=2)+'\n')

if __name__=='__main__':main()
