"""Read-only live-process/source/output forecast for the new RGB cohort."""
import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
from results.thrml_spw.execution_capacity.allocated_audit import main as earlier_audit
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'cohort_evidence'


def main():
    old = io.StringIO()
    with contextlib.redirect_stdout(old):
        earlier_audit()
    previous = json.loads(old.getvalue())
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    launch = json.loads((OUT/'launch.json').read_text())
    proc = Path('/proc')/str(launch['pid'])
    try:
        state = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
        command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
        live = int(state[19]) == launch['startticks'] and state[0] != 'Z' and 'cifar_cnn_native.cohort.pipeline' in command
    except (FileNotFoundError, ProcessLookupError):
        live, command = False, None
    assert pipeline['status'] != 'failed', pipeline
    if not live:
        assert pipeline['status'] == 'all150compactRGBnative_cases_independently_audited_goal_active'
        for path, expected in json.loads((OUT/'completion.json').read_text())['artifact_sha256'].items():
            assert sha(path) == expected, path
    produced = 0
    if (OUT/'manifest.json').exists():
        produced = len(json.loads((OUT/'manifest.json').read_text())['rows'])
    assert 0 <= produced <= 150
    future = {'native_remaining_cases': (150-produced)*1370000,
              'full_budget_GPU_preflight': 0 if (OUT/'preflight.json').exists() else 20000000,
              'controls_and_input_data': 0 if (OUT/'controls.json').exists() else 80000000,
              'statistics_support_and_logs': 0 if (OUT/'completion.json').exists() else 24500000}
    total = sum(future.values())
    assert total <= p['output_bound_bytes']
    free = shutil.disk_usage('.').free
    combined = total+previous['combined_remaining_outputs_bound_bytes']
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': sha(__file__),
              'protocol_sha256': sha(OUT/'protocol.json'), 'actually_live_parent': live, 'parent_pid': launch['pid'],
              'parent_startticks': launch['startticks'], 'parent_command': command, 'pipeline_status': pipeline['status'],
              'source_hash_checks': len(p['source_sha256']), 'input_hash_checks': len(p['input_sha256']),
              'production_cases_committed': produced, 'RGB_remaining_output_bounds_bytes': future,
              'earlier_remaining_output_bound_bytes': previous['combined_remaining_outputs_bound_bytes'],
              'combined_remaining_outputs_bound_bytes': combined, 'free_bytes_observed': free,
              'residual_after_bounds_bytes': free-combined, 'fits_remaining_output_bounds': free >= combined,
              'scope': 'Read-only process/source/input/terminal receipt and output forecast. Not physical costs or disk reservation. No source/result changes, sampling, cleanup or restart.'}
    folder = ROOT/'capacity_checks'
    folder.mkdir(exist_ok=True)
    write(folder/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
