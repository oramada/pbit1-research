"""Read-only combined forecast/liveness for word and learned RGB cohorts."""
import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
from results.thrml_spw.cifar_cnn_native.word_capacity_audit import main as prior_audit
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import ROOT, OUT, sha, write, guard, seal


def main():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        prior_audit()
    prior = json.loads(stream.getvalue())
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    launch = json.loads((OUT/'launch_receipt.json').read_text())
    assert launch['protocol_sha256'] == sha(OUT/'protocol.json')
    status = json.loads((OUT/'pipeline.json').read_text())['status']
    assert status not in ['failed', 'word_parent_terminal_before_success'], status
    try:
        base = Path('/proc')/str(launch['pid'])
        stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
        argv = (base/'cmdline').read_bytes().split(b'\0')
        live = stat[19] == launch['start_ticks'] and stat[0] not in ['Z', 'X'] and b'results.thrml_spw.cifar_cnn_native.probability_cohort.pipeline' in argv
    except (FileNotFoundError, ProcessLookupError):
        live = False
    complete = status == 'full_probability_finetuned_RGB_independently_audited_goal_active'
    if complete:
        seal(OUT/'completion.json')
    else:
        assert live, 'Learned cohort has no matching live queued/running process.'
    cases = len(json.loads((OUT/'manifest.json').read_text())['rows']) if (OUT/'manifest.json').exists() else 0
    assert 0 <= cases <= 300
    future = 0 if complete else p['storage']['all_outputs_bound']
    combined = prior['combined_remaining_outputs_bound_bytes']+future
    modules = {b'results.thrml_spw.cifar_cnn_native.cohort.run',
               b'results.thrml_spw.cifar_cnn_native.word_cohort.run',
               b'results.thrml_spw.cifar_cnn_native.probability_cohort.run'}
    jobs = []
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():
            continue
        try:
            argv = (base/'cmdline').read_bytes().split(b'\0')
            if not argv or b'.venv-thrml-gpu/bin/python' not in argv[0]:
                continue
            mi = argv.index(b'-m')+1
            if argv[mi] not in modules:
                continue
            stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] not in ['Z', 'X']:
                jobs.append({'pid': int(base.name), 'start_ticks': stat[19],
                             'module': argv[mi].decode(), 'action': argv[mi+1].decode()})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    assert len(jobs) <= 1, jobs
    free = shutil.disk_usage('.').free
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'protocol.json'),
              'earlier_remaining_outputs_bound_bytes': prior['combined_remaining_outputs_bound_bytes'],
              'learned_remaining_outputs_bound_bytes': future, 'combined_remaining_outputs_bound_bytes': combined,
              'free_bytes_observed': free, 'residual_after_bounds_bytes': free-combined,
              'fits_remaining_output_bounds': free >= combined, 'learned_pipeline_status': status,
              'learned_pipeline_live': live, 'learned_pipeline_pid': launch['pid'],
              'learned_pipeline_start_ticks': launch['start_ticks'], 'learned_cases_committed': cases,
              'word_cases_committed': prior['word_cases_committed'], 'actual_RGB_GPU_jobs': jobs,
              'source_hash_checks': len(p['source_sha256']), 'input_hash_checks': len(p['input_sha256']),
              'scope': 'Read-only actual process/source/input/terminal receipt and conservative total output forecast, not disk reservation or physical costs. No score calculation, cleanup, sampling, source mutation or restart.'}
    folder = ROOT/'probability_capacity_checks'
    folder.mkdir(exist_ok=True)
    write(folder/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
