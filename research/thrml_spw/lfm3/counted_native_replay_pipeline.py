"""Wait on original replay; test count factorization without automatic adoption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .adaptation_native_v2_pipeline import identity, read


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniform-pid', required=True, type=int)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared = root / 'counted_native_replay_protocol.json'; p = read(shared)
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items(): assert sha(Path(name)) == digest
    for name, digest in p['inputs_sha256'].items(): assert sha(root / name) == digest
    upstream = identity(args.uniform_pid)
    assert upstream is not None and upstream['state'] != 'Z'
    assert 'research.thrml_spw.lfm3.adaptation_native_v2_pipeline' in upstream['argv']
    out = root / 'counted_native_replay'; out.mkdir(exist_ok=True)
    with (out / 'pipeline_protocol.json').open('x') as f:
        json.dump({'source_sha256': sha(Path(__file__)), 'shared_protocol_sha256': sha(shared),
                   'upstream_pid': args.uniform_pid, 'upstream_identity': upstream}, f, indent=2)
    state = {'pid': os.getpid(), 'status': 'waiting_for_original_strict_replay',
             'protocol_sha256': sha(shared), 'completed': []}
    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    original_audit = root / 'adaptation_native_v2/initial_replay/independent_audit.json'
    while not original_audit.exists():
        actual = identity(args.uniform_pid)
        assert actual is not None and actual['state'] != 'Z'
        assert actual['argv'] == upstream['argv'] and actual['start_ticks'] == upstream['start_ticks']
        status = read(root / 'adaptation_native_v2/pipeline.json')
        if status['status'] == 'failed_preserved_no_restart':
            state.update(status='upstream_failed_preserved_no_restart', upstream_status=status); save()
            raise SystemExit(1)
        state['last_verified_original_pipeline_live_unix'] = time.time(); save()
        time.sleep(30)
    accepted = read(original_audit)
    assert accepted['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
    assert accepted['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    state.update(status='running', original_replay_audit_sha256=sha(original_audit)); save()
    for module in ['counted_native_replay', 'verify_counted_native_replay']:
        with (out / (module + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module,
                '--arm', 'baseline', '--seed', '202610300', '--replay'], stdout=log, stderr=log)
            state.update(current=module, child_pid=child.pid); save()
            code = child.wait()
        state['child_exit_code'] = code
        if code:
            state['status'] = 'failed_preserved_no_restart'; save()
            raise SystemExit(code)
        state['completed'].append(module); save()
    audit = read(out / 'initial_replay/independent_audit.json')
    assert audit['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    state.update(status='full_counted_original_replay_independently_verified_no_automatic_adoption',
                 current=None, child_pid=None, audit_sha256=sha(out / 'initial_replay/independent_audit.json'))
    save()


if __name__ == '__main__':
    main()
