"""Wait for verified replay; run frozen allocation; audit paired uniform study."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .adaptation_native_v2_pipeline import identity, read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniform-pid', required=True, type=int)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'allocated_native'
    out.mkdir(exist_ok=True)
    protocol_path = root / 'allocated_native_protocol.json'
    p = read(protocol_path)
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in p['inputs_sha256'].items():
        assert sha(root / name) == digest
    expected = identity(args.uniform_pid)
    assert expected is not None and expected['state'] != 'Z'
    assert 'research.thrml_spw.lfm3.adaptation_native_v2_pipeline' in expected['argv']
    with (out / 'pipeline_protocol.json').open('x') as f:
        json.dump({'uniform_identity': expected, 'uniform_pid': args.uniform_pid, 'protocol_sha256': sha(protocol_path)}, f, indent=2)
    state = {'pid': os.getpid(), 'status': 'waiting_for_strict_replay', 'completed': []}
    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    def verify_wait():
        actual = identity(args.uniform_pid)
        assert actual is not None and actual['state'] != 'Z'
        assert actual['argv'] == expected['argv'] and actual['start_ticks'] == expected['start_ticks']
        upstream = read(root / 'adaptation_native_v2/pipeline.json')
        assert upstream['status'] != 'failed_preserved_no_restart'
        state['last_verified_uniform_live_unix'] = time.time()
        state['upstream_status'] = upstream['status']; state['upstream_current'] = upstream.get('current')
        save()
    def run(module, extra, label):
        state.update(status='running', current=label)
        save()
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
            state['child_pid'] = child.pid; save()
            code = child.wait()
        state['child_exit_code'] = code
        if code:
            state['status'] = 'failed_preserved_no_restart'; save()
            raise SystemExit(code)
        state['completed'].append(label)
        state.update(current=None, child_pid=None); save()
    save()
    replay = root / 'adaptation_native_v2/initial_replay/independent_audit.json'
    while not replay.exists():
        verify_wait(); time.sleep(30)
    audit = read(replay)
    assert audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
    assert audit['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    for seed in p['seeds']:
        for module in ['allocated_native', 'verify_allocated_native']:
            run(module, ['--arm', 'mean', '--seed', str(seed)], f'{seed}_{module}')
    state['status'] = 'waiting_for_uniform_mean_cohort'; save()
    paths = [root / 'adaptation_native_v2/mean' / str(seed) / 'independent_audit.json' for seed in p['seeds']]
    while not all(path.exists() for path in paths):
        verify_wait(); time.sleep(30)
    for module in ['allocated_native_statistics', 'verify_allocated_native_statistics']:
        run(module, [], module)
    state.update(status='allocated_uniform_paired_native_study_independently_verified', statistics_audit_sha256=sha(out / 'statistics_audit.json'))
    save()


if __name__ == '__main__':
    main()
