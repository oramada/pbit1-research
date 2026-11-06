"""Wait for real upstream processes; preflight, replay, evaluate and audit once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'start_ticks': fields[19], 'state': fields[0],
                'argv': [v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v]}
    except FileNotFoundError:
        return None


def read(path):
    for _ in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            time.sleep(.1)
    raise RuntimeError(f'Could not read complete JSON state: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-pid', type=int, required=True)
    parser.add_argument('--adaptation-audit-pid', type=int, required=True)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_native_v2'
    out.mkdir(exist_ok=True)
    protocol_path = root / 'adaptation_native_v2_protocol.json'
    protocol = read(protocol_path)
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in protocol['inputs_sha256'].items():
        assert sha(root / name) == digest
    expected_identities = {}
    for pid, module in [(args.word_pid, 'word_language_pipeline'), (args.adaptation_audit_pid, 'adaptation_audit_pipeline')]:
        current = identity(pid)
        assert current is not None and current['state'] != 'Z'
        assert 'research.thrml_spw.lfm3.' + module in current['argv']
        expected_identities[pid] = current
    with (out / 'pipeline_protocol.json').open('x') as f:
        json.dump({'source_sha256': sha(Path(__file__)), 'shared_protocol_sha256': sha(protocol_path),
                   'upstream_identities': expected_identities}, f, indent=2)
    state = {'pid': os.getpid(), 'status': 'waiting_for_existing_word_cohort', 'completed': [],
             'protocol_sha256': sha(protocol_path)}
    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    def live(pid):
        actual = identity(pid)
        expected = expected_identities[pid]
        assert actual is not None and actual['state'] != 'Z'
        assert actual['start_ticks'] == expected['start_ticks'] and actual['argv'] == expected['argv']
        state['last_verified_upstream_live_unix'] = time.time()
    def run(module, extra, label):
        state.update(status='running', current=label)
        save()
        log = (out / (label + '.log')).open('x')
        child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
        state['child_pid'] = child.pid
        save()
        code = child.wait()
        log.close()
        state['child_exit_code'] = code
        if code:
            state['status'] = 'failed_preserved_no_restart'
            save()
            raise SystemExit(code)
        state['completed'].append(label)
        state.update(current=None, child_pid=None)
        save()
    save()
    while True:
        word = read(root / 'word_language/pipeline.json')
        if word['status'] == 'all_five_word_seeds_independently_verified':
            break
        assert word['status'] in ['running', 'waiting_for_single_cohort_and_analysis']
        live(args.word_pid)
        state.update(word_completed_seeds=word['completed_seeds'], word_current_seed=word.get('current_seed'))
        save()
        time.sleep(30)
    for module in ['preflight_program_export_gpu', 'verify_program_export_gpu']:
        run(module, [], module)
    state['status'] = 'waiting_for_adaptation_audits'
    save()
    while True:
        upstream = read(root / 'adaptation_pilot/audit_pipeline.json')
        if upstream['status'] == 'all_three_arms_independently_verified_pending_native_quality':
            break
        assert upstream['status'] in ['waiting_on_verified_producer', 'auditing']
        live(args.adaptation_audit_pid)
        state['audited_adaptation_arms'] = upstream['audited_arms']
        save()
        time.sleep(30)
    # Selection is frozen from prior Torch validation, before any native
    # candidate evaluation; all three candidates are nevertheless retained.
    scores, inputs = {}, {}
    for arm in protocol['arms']:
        path = root / 'adaptation_pilot' / arm / 'results.json'
        report = read(path)
        values = [v['mean_NLL_FP64'] for v in report['after_binomial']]
        assert len(values) == 5
        scores[arm] = sum(values) / 5
        inputs[str(path)] = sha(path)
    selected = min(protocol['arms'], key=lambda arm: scores[arm])
    with (out / 'surrogate_selection.json').open('x') as f:
        json.dump({'selected_arm': selected, 'Torch_binomial_validation_NLL': scores,
                   'inputs_sha256': inputs, 'recorded_unix': time.time(),
                   'scope': 'Prior surrogate-only selection on same reserved validation data; all native candidate results retained. Not fresh-test selection or a confirmatory claim.'}, f, indent=2)
    for module in ['adaptation_native_v2', 'verify_adaptation_native_v2']:
        run(module, ['--arm', 'baseline', '--seed', '202610300', '--replay'], 'initial_replay_' + module)
    assert read(out / 'initial_replay/independent_audit.json')['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            for module in ['adaptation_native_v2', 'verify_adaptation_native_v2']:
                run(module, ['--arm', arm, '--seed', str(seed)], f'{arm}_{seed}_{module}')
    for module in ['adaptation_native_v2_statistics', 'verify_adaptation_native_v2_statistics']:
        run(module, [], module)
    state.update(status='all_native_validation_candidates_and_statistics_independently_verified',
                 statistics_audit_sha256=sha(out / 'statistics_audit.json'))
    save()


if __name__ == '__main__':
    main()
