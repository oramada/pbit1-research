"""One prospective full-model implementation replay, without retry/adoption."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol_path = root / 'batched_native_replay_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in protocol['inputs_sha256'].items():
        assert sha(root / name) == digest
    out = root / 'batched_native_replay'; out.mkdir(exist_ok=True)
    state = {'pid': os.getpid(), 'status': 'running', 'protocol_sha256': sha(protocol_path), 'completed': []}
    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    with (out / 'pipeline_protocol.json').open('x') as f:
        json.dump({'source_sha256': sha(Path(__file__)), 'shared_protocol_sha256': sha(protocol_path)}, f, indent=2)
    save()
    for module in ['batched_native_replay', 'verify_batched_native_replay']:
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
    audit = json.loads((out / 'initial_replay/independent_audit.json').read_text())
    assert audit['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    state.update(status='full_batched_original_replay_independently_verified_no_automatic_adoption',
                 current=None, child_pid=None, audit_sha256=sha(out / 'initial_replay/independent_audit.json'))
    save()


if __name__ == '__main__':
    main()
