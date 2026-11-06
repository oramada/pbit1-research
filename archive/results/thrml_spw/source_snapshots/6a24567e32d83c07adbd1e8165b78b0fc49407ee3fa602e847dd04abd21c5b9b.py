"""Execute frozen five-seed cohort sequentially; never silently restart a seed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol_path = root / 'language_native_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    controls = json.loads((root / 'language_controls/independent_audit.json').read_text())
    assert controls['status'] == 'all_saved_token_losses_and_unique_tensor_coverage_verified'
    assert sha(root / 'language_controls/independent_audit.json') == protocol['controls_audit_sha256']
    assert json.loads((root / 'language_batch_preflight/result.json').read_text())['exact_batch_vs_individual']
    out = root / 'language_native'
    out.mkdir(exist_ok=True)
    state = {'protocol_sha256': sha(protocol_path), 'pid': os.getpid(),
             'completed_seeds': [], 'status': 'running'}

    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')

    for seed in protocol['seeds']:
        assert not (out / str(seed) / 'protocol.json').exists(), 'Prior seed exists; no automatic restart.'
        state.update(current_seed=seed, stage='native_forward')
        save()
        for module, suffix in [('language_native', 'native'), ('verify_language_native', 'audit')]:
            state['stage'] = suffix
            save()
            with (out / f'{seed}_{suffix}.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module,
                                         '--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = proc.pid
                save()
                code = proc.wait()
            if code:
                state.update(status='failed_preserved_no_restart', returncode=code)
                save()
                raise RuntimeError(f'{module} seed{seed} failed; log preserved')
        audit = json.loads((out / str(seed) / 'independent_audit.json').read_text())
        assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
        state['completed_seeds'].append(seed)
        save()
    state.update(status='all_five_seeds_independently_verified', stage='complete')
    save()


if __name__ == '__main__':
    main()
