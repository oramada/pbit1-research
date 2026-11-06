"""Read-only live identity, integrity and capacity checks for the400case study."""
import datetime
import json
from pathlib import Path
import shutil
from results.thrml_spw.cifar_cnn_native.learned_word_cohort.common import (
    ROOT, OUT, sha, guard, seal, closed, gpu_jobs)


def main():
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    launch = json.loads((ROOT/'learned_word_cohort_launch.json').read_text())
    handle = next(h for h in launch['handles'] if h['argv'][2] ==
        'results.thrml_spw.cifar_cnn_native.learned_word_cohort.pipeline')
    state = json.loads((OUT/'pipeline.json').read_text())
    live = False; proc = Path('/proc')/str(handle['pid'])
    if proc.exists():
        try:
            stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            argv = [a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            live = stat[0] not in ['Z', 'X'] and stat[19] == handle['start_ticks'] and argv == handle['argv']
        except (FileNotFoundError, ProcessLookupError):
            pass
    if state['status'] == 'running':
        assert live, 'Original pipeline handle missing; inspect terminal/log state, do not automatically restart'
    elif state['status'] == 'all400_samefit_word_cases_independently_verified_goal_active':
        seal(OUT/'completion.json')
    else:
        assert state['status'] == 'failed', state['status']
    jobs = gpu_jobs(); assert len(jobs) <= 1
    for job in jobs:
        assert job['module'] == 'results.thrml_spw.cifar_cnn_native.learned_word_cohort.run'
        stat = (Path('/proc')/str(job['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        assert stat[19] == job['start_ticks'] and stat[0] not in ['Z', 'X']
    expected = [(m, mi, t, seed) for m in range(2) for mi in range(5)
        for t in p['word_samples'] for seed in range(10)]
    native_cases = replay_cases = scheduled = 0
    if (OUT/'manifest.json').exists():
        m = json.loads((OUT/'manifest.json').read_text()); native_cases = len(m['rows'])
        assert [(r['method'], r['model'], r['T'], r['seed']) for r in m['rows']] == expected[:native_cases]
        scheduled = sum(1250*15712*r['T'] for r in m['rows'])
        assert m['scheduled_updates'] == scheduled and m['counter_entries'] == native_cases*61582500
        assert m['protocol_sha256'] == sha(OUT/'protocol.json')
        assert all(r['matched_single_T'] == 4*r['T'] for r in m['rows'])
        if m['complete']:
            assert native_cases == 400 and scheduled == p['retained_updates_per_pass']
    if (OUT/'audit_progress.json').exists():
        a = json.loads((OUT/'audit_progress.json').read_text()); replay_cases = len(a['rows'])
        assert [(r['method'], r['model'], r['T'], r['seed']) for r in a['rows']] == expected[:replay_cases]
        assert a['counter_entries'] == replay_cases*61582500
        assert a['scheduled_updates'] == sum(1250*15712*r['T'] for r in a['rows'])
        assert all(r['all_receipts_and_final_logits_exact'] for r in a['rows'])
    complete = (OUT/'completion.json').exists()
    bound = 0 if complete else p['output_bounds_bytes']['total']
    free = shutil.disk_usage('.').free; assert free > bound
    now = datetime.datetime.now(datetime.timezone.utc)
    result = {'recorded_utc': now.isoformat(), 'source_sha256': sha(__file__),
        'protocol_sha256': sha(OUT/'protocol.json'), 'pipeline_status': state['status'],
        'pipeline_pid': handle['pid'], 'pipeline_start_ticks': handle['start_ticks'], 'pipeline_live': live,
        'native_cases_committed': native_cases, 'replay_cases_verified': replay_cases,
        'native_retained_updates_committed': scheduled, 'actual_scientific_GPU_workers': jobs,
        'source_hash_checks': len(p['source_sha256']), 'input_hash_checks': len(p['input_sha256']),
        'remaining_outputs_bound_bytes': bound, 'free_bytes_observed': free, 'fits_output_bounds': True,
        'scope': 'Actual PID/start-tick/argv, source/input/prerequisite closure, committed prefix/work and conservative capacity audit; no partial task scoring/restart/cleanup. Writes monitoring receipts; do not run during inventory refresh/verification.'}
    dest = ROOT/'learned_word_capacity_checks'; dest.mkdir(exist_ok=True)
    (dest/f'{now.strftime("%Y%m%dT%H%M%SZ")}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
