"""Combined read-only capacity/liveness check including the queued RGB word cohort."""
import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
from results.thrml_spw.cifar_cnn_native.capacity_audit import main as single_audit
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'word_cohort_evidence'


def main():
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        single_audit()
    single = json.loads(captured.getvalue())
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    launch = json.loads((OUT/'launch_receipt.json').read_text())
    assert launch['protocol_sha256'] == sha(OUT/'protocol.json')
    status = json.loads((OUT/'pipeline.json').read_text())
    assert status['status'] not in ['failed', 'parent_terminal_before_audit_success'], status
    try:
        base = Path('/proc')/str(launch['pid'])
        stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
        argv = (base/'cmdline').read_bytes().split(b'\0')
        live = stat[19] == launch['start_ticks'] and stat[0] not in ['Z', 'X'] and b'results.thrml_spw.cifar_cnn_native.word_cohort.pipeline' in argv
    except (FileNotFoundError, ProcessLookupError):
        live = False
    complete = status['status'] == 'all150RGBword_cases_independently_audited_goal_active'
    if complete:
        for path, expected in json.loads((OUT/'completion.json').read_text())['artifact_sha256'].items():
            assert sha(path) == expected, path
    else:
        assert live, 'Queued/running word pipeline has no matching live process.'
    produced = 0
    if (OUT/'manifest.json').exists():
        manifest = json.loads((OUT/'manifest.json').read_text())
        assert manifest['protocol_sha256'] == sha(OUT/'protocol.json')
        produced = len(manifest['rows'])
        assert 0 <= produced <= 150
    # Keep the whole word output allowance until terminal closure. Conservative:
    # already written word files are still included in this future allowance.
    word_bound = 0 if complete else p['output_bound_bytes']
    combined = single['combined_remaining_outputs_bound_bytes']+word_bound
    actual_gpu_jobs = []
    modules = {b'results.thrml_spw.cifar_cnn_native.cohort.run',
               b'results.thrml_spw.cifar_cnn_native.word_cohort.run'}
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():
            continue
        try:
            argv = (base/'cmdline').read_bytes().split(b'\0')
            if not argv or b'.venv-thrml-gpu/bin/python' not in argv[0]:
                continue
            module_index = argv.index(b'-m')+1
            if argv[module_index] not in modules:
                continue
            stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] in ['Z', 'X']:
                continue
            actual_gpu_jobs.append({'pid': int(base.name), 'start_ticks': stat[19],
                                    'module': argv[module_index].decode(),
                                    'action': argv[module_index+1].decode()})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    assert len(actual_gpu_jobs) <= 1, actual_gpu_jobs
    free = shutil.disk_usage('.').free
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'word_protocol_sha256': sha(OUT/'protocol.json'),
              'single_and_earlier_remaining_output_bound_bytes': single['combined_remaining_outputs_bound_bytes'],
              'word_remaining_output_bound_bytes': word_bound,
              'combined_remaining_outputs_bound_bytes': combined, 'free_bytes_observed': free,
              'residual_after_bounds_bytes': free-combined, 'fits_remaining_output_bounds': free >= combined,
              'word_pipeline_live': live, 'word_pipeline_pid': launch['pid'],
              'word_pipeline_start_ticks': launch['start_ticks'], 'word_pipeline_status': status['status'],
              'word_cases_committed': produced, 'single_cases_committed': single['production_cases_committed'],
              'actual_RGB_GPU_processes': actual_gpu_jobs,
              'word_source_hash_checks': len(p['source_sha256']), 'word_input_hash_checks': len(p['input_sha256']),
              'scope': 'Read-only sources/processes/terminal receipts and conservative combined output forecast. No task scoring, sampling, source edits, cleanup or restart. This does not reserve disk space or measure hardware cost.'}
    folder = ROOT/'combined_capacity_checks'
    folder.mkdir(exist_ok=True)
    write(folder/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
