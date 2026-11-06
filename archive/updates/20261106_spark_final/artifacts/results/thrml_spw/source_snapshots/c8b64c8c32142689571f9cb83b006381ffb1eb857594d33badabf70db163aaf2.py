"""Read-only capacity/source/process audit including both allocation queues."""
from pathlib import Path
import contextlib
import datetime
import hashlib
import io
import json
import shutil

from results.thrml_spw.execution_capacity import audit as legacy

ROOT = Path('results/thrml_spw')
NATIVE = ROOT/'fashion_cnn_native/allocated_native_evidence'
STATS = ROOT/'fashion_cnn_native/allocated_statistics_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role(folder, command_fragment, terminal_status, completion):
    launch = json.loads((folder/'launch.json').read_text())
    state = json.loads((folder/'pipeline.json').read_text())
    p = json.loads((folder/'protocol.json').read_text())
    checked = 0
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path
            checked += 1
    proc = Path('/proc')/str(launch['pid'])
    try:
        stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
        command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
        live = int(stat[19]) == launch['startticks'] and stat[0] != 'Z' and command_fragment in command
    except (FileNotFoundError, ProcessLookupError):
        live, command = False, None
    assert state['status'] not in ['failed', 'prior_pipeline_terminal_without_success', 'native_parent_terminal_without_success'], state
    receipt = None
    if not live:
        state = json.loads((folder/'pipeline.json').read_text())
        assert state['status'] == terminal_status, (folder, state)
        assert completion.exists()
        receipt = json.loads(completion.read_text())
        if 'artifact_sha256' in receipt:
            for path, expected in receipt['artifact_sha256'].items():
                assert sha(path) == expected, path
        else:
            assert receipt['passed'] and receipt['complete'] and receipt['manifest_sha256'] == sha(folder/'manifest.json')
        for stage in state['stages']:
            assert stage['status'] == 'completed' and stage['returncode'] == 0
            assert sha(folder/'logs'/(stage['stage']+'.log')) == stage['log_sha256']
    return {'pid': launch['pid'], 'expected_startticks': launch['startticks'], 'actually_live': live,
            'pipeline_status': state['status'], 'command': command,
            'protocol_sha256': sha(folder/'protocol.json'), 'source_and_input_hash_checks': checked,
            'terminal_receipt_sha256': sha(completion) if receipt is not None else None}


def main():
    # Keep the old helper immutable; its forecast excludes allocation queues.
    assert sha(legacy.__file__) == 'f6467e6d153c7ac4156321ddb2bebd072865fdb72735398774ce124822bd8d89'
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        legacy.main()
    old = json.loads(captured.getvalue())
    roles = {'allocated_native': role(NATIVE, 'fashion_cnn_native.allocated_native.pipeline',
                                      'all100native_allocated_comparisons_publicly_replayed_statistics_pending_goal_active', NATIVE/'native_audit.json'),
             'allocated_statistics': role(STATS, 'fashion_cnn_native.allocated_statistics.pipeline',
                                          'all100matched_budget_CNN_cases_statistically_closed_goal_active', STATS/'completion.json')}
    count = 0
    if (NATIVE/'manifest.json').exists():
        m = json.loads((NATIVE/'manifest.json').read_text())
        count = len(m['rows'])
        assert 0 <= count <= 100
    if (STATS/'completion.json').exists():
        future = {'native_remaining_cases': 0, 'native_preflight': 0, 'statistics_and_support': 0}
    else:
        future = {'native_remaining_cases': (100-count)*1370000,
                  'native_preflight': 0 if (NATIVE/'preflight.json').exists() else 20000000,
                  'statistics_and_support': 18000000}
    additional = sum(future.values())
    assert additional <= json.loads((NATIVE/'protocol.json').read_text())['all_future_outputs_bound_bytes']
    free = shutil.disk_usage('.').free
    total = old['remaining_outputs_total_bound_bytes']+additional
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': sha(__file__),
              'legacy_audit_source_sha256': old['source_sha256'], 'legacy_sealed_source_hash_checks': old['sealed_source_hash_checks'],
              'legacy_remaining_output_bound_bytes': old['remaining_outputs_total_bound_bytes'],
              'new_live_or_terminal_receipt_checked_roles': roles, 'allocated_cases_committed': count,
              'allocated_remaining_output_bounds_bytes': future, 'combined_remaining_outputs_bound_bytes': total,
              'free_bytes_observed': free, 'residual_after_bounds_bytes': free-total,
              'fits_full_queued_native_guard_now': free-old['remaining_outputs_total_bound_bytes'] >= 190000000,
              'fits_full_remaining_output_bounds_now': free >= total,
              'scope': 'Read-only source/input/process/terminal-receipt audit and conservative output payload forecast including both allocation queues. Not a disk reservation, external-writer bound or physical resource claim; no source/experiment changes, restarts, cache cleanup or native draws.'}
    folder = ROOT/'execution_capacity'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ_allocated')
    folder.mkdir()
    (folder/'legacy_audit_capture.json').write_text(json.dumps(old, indent=2)+'\n')
    (folder/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
