"""Five fixed mean-program seeds and audits; preserve all outcomes, no retries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); shared = root / 'counted_mean_validation_protocol.json'
    p = json.loads(shared.read_text())
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items(): assert sha(Path(name)) == digest
    for name, digest in p['inputs_sha256'].items(): assert sha(root / name) == digest
    assert json.loads((root / 'counted_native_replay/initial_replay/independent_audit.json').read_text())['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    assert json.loads((root / 'adaptation_native_v2/surrogate_selection.json').read_text())['selected_arm'] == 'mean'
    out = root / 'counted_mean_validation'; out.mkdir(exist_ok=True)
    state = {'pid': os.getpid(), 'status': 'running', 'completed': [], 'protocol_sha256': sha(shared)}
    with (out / 'pipeline_protocol.json').open('x') as f: json.dump({'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(shared)}, f, indent=2)
    def save(): (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    def run(module, extra, label):
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
            state.update(current=label, child_pid=child.pid); save()
            code = child.wait()
        state['child_exit_code'] = code
        if code:
            state['status'] = 'failed_preserved_no_restart'; save(); raise SystemExit(code)
        state['completed'].append(label); save()
    save()
    for seed in p['seeds']:
        for module in ['counted_mean_validation', 'verify_counted_mean_validation']:
            run(module, ['--arm', 'mean', '--seed', str(seed)], f'{seed}_{module}')
    for module in ['counted_mean_statistics', 'verify_counted_mean_statistics']: run(module, [], module)
    state.update(status='all_five_counted_mean_cases_and_statistics_independently_verified', current=None, child_pid=None,
                 statistics_audit_sha256=sha(out / 'statistics_audit.json')); save()


if __name__ == '__main__':
    main()
