"""Combined old-cohort closure and actual T4096 process/output monitoring."""
import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
from results.thrml_spw.cifar_cnn_native.probability_capacity_audit import main as previous_audit
from results.thrml_spw.cifar_cnn_native.probability_high_budget.common import (
    ROOT, OUT, sha, write, guard, seal)
from results.thrml_spw.cifar_cnn_native.probability_high_budget.pipeline import gpu_jobs, SUCCESS


def main():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        previous_audit()
    earlier = json.loads(stream.getvalue())
    assert earlier['combined_remaining_outputs_bound_bytes'] == 0
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    launch = json.loads((OUT/'launch_receipt.json').read_text())
    assert launch['protocol_sha256'] == sha(OUT/'protocol.json')
    state = json.loads((OUT/'pipeline.json').read_text())
    assert state['status'] != 'failed', state
    try:
        proc = Path('/proc')/str(launch['pid'])
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        argv = (proc/'cmdline').read_bytes().split(b'\0')
        live = stat[19] == launch['start_ticks'] and stat[0] not in ['Z', 'X'] and b'results.thrml_spw.cifar_cnn_native.probability_high_budget.pipeline' in argv
    except (FileNotFoundError, ProcessLookupError):
        live = False
    complete = state['status'] == SUCCESS
    if complete:
        seal(OUT/'completion.json')
    else:
        assert live, 'No matching live high-budget pipeline; inspect original handle/state without automatic restart'
    manifest = json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {'rows': [], 'native_updates': 0}
    expected = [(method, model, 4096, seed) for method in range(2) for model in range(5) for seed in range(10)]
    rows = manifest['rows']
    assert [(r['method'], r['model'], r['T'], r['seed']) for r in rows] == expected[:len(rows)] and len(rows) <= 100
    assert manifest['native_updates'] == sum(r['native_updates'] for r in rows) == len(rows)*1250*3928*4096
    jobs = gpu_jobs()
    assert len(jobs) <= 1, jobs
    remaining = 0 if complete else p['storage']['all_outputs_bound']
    free = shutil.disk_usage('.').free
    assert free >= remaining
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'protocol.json'),
        'earlier_remaining_outputs_bound_bytes': 0, 'high_budget_remaining_outputs_bound_bytes': remaining,
        'combined_remaining_outputs_bound_bytes': remaining, 'free_bytes_observed': free,
        'fits_remaining_output_bounds': free >= remaining, 'pipeline_status': state['status'],
        'pipeline_pid': launch['pid'], 'pipeline_start_ticks': launch['start_ticks'], 'pipeline_live': live,
        'cases_committed': len(rows), 'retained_updates_committed': manifest['native_updates'],
        'actual_scientific_GPU_workers': jobs, 'source_hash_checks': len(p['source_sha256']),
        'input_hash_checks': len(p['input_sha256']),
        'scope': 'Actual PID/start-tick/argv/source/input/closure and fixed partial schedule verification. No partial task metrics, restart, cleanup or measured physical costs. Writes monitoring receipts; do not run during inventory verification.'}
    folder = ROOT/'high_budget_capacity_checks'
    folder.mkdir(exist_ok=True)
    write(folder/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
