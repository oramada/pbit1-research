"""Run the fixed K8 baseline cohort and independent audits once, without retry."""
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
    protocol_path = root / 'word8_language_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['prerequisites_sha256'].items():
        assert sha(root / name) == digest
    assert json.loads((root / 'word8_preflight/independent_audit.json').read_text())['status'] == 'all16_cases_signs_public_counts_integer_accumulators_and_readouts_verified'
    out = root / 'word8_language'
    out.mkdir(exist_ok=True)
    state = {'status': 'running', 'pid': os.getpid(), 'protocol_sha256': sha(protocol_path), 'completed_seeds': []}
    path = out / 'pipeline.json'
    with path.open('x') as f:
        json.dump(state, f, indent=2)
    def save():
        path.write_text(json.dumps(state, indent=2) + '\n')
    def run(module, extra, label):
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
            state.update(stage=label, child_pid=child.pid)
            save()
            code = child.wait()
        if code:
            state.update(status='failed_preserved_no_restart', child_exit_code=code)
            save()
            raise SystemExit(code)
    for seed in protocol['seeds']:
        assert not (out / str(seed) / 'protocol.json').exists()
        state['current_seed'] = seed
        for module in ['word8_language', 'verify_word8_language']:
            run(module, ['--seed', str(seed)], str(seed) + '_' + module)
        audit = json.loads((out / str(seed) / 'independent_audit.json').read_text())
        assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
        assert audit['retained_work_matches_paired_single_T1024'] and audit['retained_work_matches_paired_word4_T256']
        assert audit['warmup_eight_times_paired_single'] and audit['warmup_twice_paired_word4']
        state['completed_seeds'].append(seed)
        save()
    state.update(status='all_five_word8_seeds_independently_verified', current_seed=None)
    save()
    for module in ['word8_language_statistics', 'verify_word8_language_statistics']:
        run(module, [], module)
    state.update(status='all_K8_seeds_and_three_method_statistics_independently_verified', child_pid=None,
                 statistics_audit_sha256=sha(out / 'statistics_audit.json'))
    save()


if __name__ == '__main__':
    main()
