"""Read-only job/source/capacity check; no sampling, cleanup or restarts."""
from pathlib import Path
import json,hashlib,datetime,shutil
ROOT=Path('results/thrml_spw');OUT=ROOT/'execution_capacity'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=OUT/stamp;folder.mkdir()
    paths={'integer':ROOT/'integer_activation_native/cohort','CNN_GPU':ROOT/'fashion_cnn_native/throughput_evidence','CNN':ROOT/'fashion_cnn_native/cohort_evidence','precision':ROOT/'fashion_cnn_native/precision_evidence'}
    roles={'integer':(3778471,'44679072'),'CNN_GPU':(3790833,'44969679'),'CNN':(3792481,'45024359'),'precision':(3794515,'45061769')};states={};seals=0
    word=ROOT/'fashion_cnn_native/word_comparison_evidence'
    if (word/'pipeline.json').exists():
        paths['word']=word;matches=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
                if cmd.startswith('.venv-thrml/bin/python -m results.thrml_spw.fashion_cnn_native.word_comparison.pipeline'):
                    matches.append((int(proc.name),proc.joinpath('stat').read_text().split()[21]))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        assert len(matches)<=1,('Duplicate word runners',matches)
        roles['word']=matches[0] if matches else (-1,'terminal')
    probability_native=ROOT/'fashion_cnn_native/probability_native_evidence'
    if (probability_native/'pipeline.json').exists():
        paths['probability_native']=probability_native;matches=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
                if cmd.startswith('.venv/bin/python -m results.thrml_spw.fashion_cnn_native.probability_native.watch run'):
                    matches.append((int(proc.name),proc.joinpath('stat').read_text().split()[21]))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        assert len(matches)<=1,('Duplicate learned-probability validation watchers',matches)
        roles['probability_native']=matches[0] if matches else (-1,'terminal')
    probability_cohort=ROOT/'fashion_cnn_native/probability_cohort_evidence'
    if (probability_cohort/'pipeline.json').exists():
        paths['probability_cohort']=probability_cohort;matches=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
                if cmd.startswith('.venv/bin/python -m results.thrml_spw.fashion_cnn_native.probability_cohort.pipeline run'):
                    matches.append((int(proc.name),proc.joinpath('stat').read_text().split()[21]))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        assert len(matches)<=1,('Duplicate learned-probability full-cohort runners',matches)
        roles['probability_cohort']=matches[0] if matches else (-1,'terminal')
    for role,path in paths.items():
        pipeline=json.loads((path/'pipeline.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
        for file,h in protocol['source_sha256'].items():assert sha(file)==h,file;seals+=1
        pid,ticks=roles[role];proc=Path('/proc')/str(pid);live=proc.exists() and proc.joinpath('stat').read_text().split()[21]==ticks
        if not live:
            pipeline=json.loads((path/'pipeline.json').read_text())
            assert pipeline['status'].endswith('goal_active') and (path/'completion.json').exists(),(role,pipeline)
            for file,h in json.loads((path/'completion.json').read_text())['artifact_sha256'].items():assert sha(file)==h,file
        states[role]={'pid':pid,'expected_start_ticks':ticks,'actually_live':live,'pipeline_status':pipeline['status']}
    captured=(paths['integer']/'manifest.json').read_bytes();(folder/'integer_manifest_capture.json').write_bytes(captured);manifest=json.loads(captured);n=len(manifest['rows']);assert 0<=n<=150
    # ZIP/NPY overhead is included in per-case bounds, beyond exact raw payload.
    integer_remaining=(150-n)*3020000
    analysis=0 if (paths['integer']/'statistics_audit.json').exists() else 22000000
    gpu=0 if (paths['CNN_GPU']/'completion.json').exists() else 5000000
    cnn_controls=0 if (paths['CNN']/'controls.json').exists() else 50000000
    cn=0
    if (paths['CNN']/'manifest.json').exists():cn=len(json.loads((paths['CNN']/'manifest.json').read_text())['rows'])
    cnn_remaining=(150-cn)*1370000+(0 if (paths['CNN']/'statistics_audit.json').exists() else 3000000)
    precision=0 if (paths['precision']/'completion.json').exists() else 18000000
    word_cases=0;word_remaining=0
    if 'word' in paths:
        if (word/'manifest.json').exists():word_cases=len(json.loads((word/'manifest.json').read_text())['rows'])
        word_remaining=(150-word_cases)*1370000+(0 if (word/'completion.json').exists() else 12000000)
    probability_cases=0;probability_remaining=0
    if 'probability_native' in paths:
        if (probability_native/'progress.json').exists():probability_cases=len(json.loads((probability_native/'progress.json').read_text())['rows'])
        probability_remaining=0 if (probability_native/'completion.json').exists() else (30-probability_cases)*550000+3500000
    fine_cases=0;fine_remaining=0;fine_controls=0
    if 'probability_cohort' in paths:
        if (probability_cohort/'manifest.json').exists():fine_cases=len(json.loads((probability_cohort/'manifest.json').read_text())['rows'])
        fine_controls=0 if (probability_cohort/'controls.json').exists() else 8000000
        fine_remaining=0 if (probability_cohort/'completion.json').exists() else (300-fine_cases)*1370000+fine_controls+11000000
    free=shutil.disk_usage('.').free;forecast={'integer_remaining_cases':integer_remaining,'integer_controls_statistics':analysis,'CNN_GPU_validation':gpu,'CNN_controls_data':cnn_controls,'CNN_remaining_cases_statistics':cnn_remaining,'precision_controls_statistics':precision,'word_comparison_cases_and_support':word_remaining,'learned_probability_native_validation':probability_remaining,'learned_probability_full_cohort':fine_remaining};total=sum(forecast.values())
    precontrols=free-integer_remaining-analysis-gpu;prenative=precontrols-cnn_controls-precision
    before_word=free-integer_remaining-analysis-gpu-cnn_controls-cnn_remaining-precision
    result={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'captured_integer_manifest_sha256':hashlib.sha256(captured).hexdigest(),'integer_cases_committed':n,'CNN_cases_committed':cn,'word_cases_committed':word_cases,'live_or_terminal_receipt_checked_roles':states,'sealed_source_hash_checks':seals,'free_bytes_observed':free,'remaining_output_bounds_bytes':forecast,'remaining_outputs_total_bound_bytes':total,'residual_after_bounds_bytes':free-total,'projected_free_before_CNN_controls_bytes':precontrols,'projected_free_before_CNN_native_bytes':prenative,'projected_free_before_word_stages_bytes':before_word,'CNN_controls_guard_bytes':450000000,'CNN_native_guard_bytes':400000000,'word_stages_guard_bytes':450000000,'fits_word_guard_now':before_word>=450000000,'fits_controls_guard_now':precontrols>=450000000,'fits_native_guard_now':prenative>=400000000,'scope':'Conservative file-payload forecast from immutable captured rows plus protocol array sizes/ZIP overhead/reserves, not a reservation or guarantee against external disk writers. In-flight temporary files/finished partial downstream files may be double charged conservatively. Existing runtime guards remain authoritative. No files removed, jobs restarted or native draws added.'}
    before_fine=free-(total-fine_remaining)
    result.update(learned_probability_full_cases_committed=fine_cases,projected_free_before_learned_probability_controls_bytes=before_fine,projected_free_before_learned_probability_native_bytes=before_fine-fine_controls,learned_probability_controls_guard_bytes=480000000,learned_probability_native_guard_bytes=472000000,fits_learned_probability_controls_guard=before_fine>=480000000,fits_learned_probability_native_guard=before_fine-fine_controls>=472000000)
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
