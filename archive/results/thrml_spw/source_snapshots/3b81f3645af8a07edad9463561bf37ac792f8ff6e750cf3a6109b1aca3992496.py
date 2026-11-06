"""Wait for an existing model slot, then evaluate both matched spatial controls.

No retries, score-based stopping, or mutation of existing one-p-bit studies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .adaptation_native_v2_pipeline import identity, read
from .count_budget_continuation import needs_higher_budget


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_completion(state, audit, stats_sha, audit_sha, expected_state,
                        expected_audit, expected_protocol, expected_verifier):
    assert state['status'] == expected_state
    assert state['protocol_sha256'] == expected_protocol
    assert state['statistics_audit_sha256'] == audit_sha
    assert audit['status'] == expected_audit
    assert audit['source_sha256'] == expected_verifier
    assert audit['statistics_sha256'] == stats_sha
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-pid', type=int, required=True)
    parser.add_argument('--single-pid', type=int, required=True)
    parser.add_argument('--higher-pid', type=int, required=True)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared = root / 'adapted_word_validation_protocol.json'
    p = read(shared)
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in p['inputs_sha256'].items():
        assert sha(root / name) == digest
    out = root / 'adapted_word_validation'
    out.mkdir(exist_ok=False)
    specifications = {
        'word8_language': (args.word_pid, 'word8_language_pipeline', 'all_K8_seeds_and_three_method_statistics_independently_verified',
                          'all_three_methods_fifteen_contrasts_and_matched_resources_verified'),
        'counted_mean_validation': (args.single_pid, 'counted_mean_validation_pipeline', 'all_five_counted_mean_cases_and_statistics_independently_verified',
                                    'all_five_mean_quality_statistics_and_workspace_resource_counts_verified'),
        'counted_mean_t4096': (args.higher_pid, 'counted_mean_t4096_pipeline', 'all_five_conditional_T4096_cases_and_statistics_independently_verified',
                              'all_five_mean_quality_statistics_and_workspace_resource_counts_verified'),
    }
    identities = {name: identity(spec[0]) for name, spec in specifications.items()}
    for name, spec in specifications.items():
        current = read(root / name / 'pipeline.json')
        terminal = [spec[2]]
        if name == 'counted_mean_t4096':
            terminal.append('extension_not_executed_lower_budget_quality_gate_passed')
        if current['status'] not in terminal:
            observed = identities[name]
            assert observed and observed['state'] != 'Z'
            assert 'research.thrml_spw.lfm3.' + spec[1] in observed['argv']
    with (out / 'pipeline_protocol.json').open('x') as handle:
        json.dump({'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(shared),
                   'upstream_identities': identities, 'minimum_available_host_kB': p['minimum_available_host_kB']}, handle, indent=2)
    state = {'pid': os.getpid(), 'status': 'waiting_for_original_K8_and_all_single_T1024_seeds',
             'protocol_sha256': sha(shared), 'completed': [], 'completed_budgets': []}
    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    def live(name):
        actual, expected = identity(specifications[name][0]), identities[name]
        assert actual and expected and actual['state'] != 'Z'
        assert actual['start_ticks'] == expected['start_ticks'] and actual['argv'] == expected['argv']
        state['last_verified_upstream_live_unix'] = time.time()
    def ready(name):
        current = read(root / name / 'pipeline.json')
        assert current['protocol_sha256'] == p['upstream_protocols_sha256'][name]
        assert 'failed' not in current['status']
        if current['status'] != specifications[name][2]:
            live(name)
            return False
        audit_path, stats_path = root / name / 'statistics_audit.json', root / name / 'statistics.json'
        return validate_completion(current, read(audit_path), sha(stats_path), sha(audit_path),
            specifications[name][2], specifications[name][3], p['upstream_protocols_sha256'][name],
            p['upstream_statistics_verifiers_sha256'][name])
    def resources():
        while True:
            available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            fs = os.statvfs(root)
            free_disk = fs.f_bavail * fs.f_frsize
            state.update(available_host_kB=available, free_disk_bytes=free_disk)
            if available >= p['minimum_available_host_kB'] and free_disk >= p['minimum_free_disk_bytes']:
                return
            state.update(status='waiting_for_memory_or_disk_headroom')
            save()
            time.sleep(30)
    def run(module, extra, label):
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
            state.update(status='running', current=label, child_pid=child.pid)
            save()
            code = child.wait()
        assert code == 0, f'{label} exited {code}; no retry'
        state['completed'].append(label)
        state.update(child_pid=None)
        save()
    def evaluate_budget(budget):
        for bits in [4, 8]:
            for seed in p['seeds_by_budget'][str(budget)]:
                resources()
                extra = ['--arm', 'mean', '--seed', str(seed), '--budget', str(budget), '--word-bits', str(bits)]
                for module in ['adapted_word_validation', 'verify_adapted_word_validation']:
                    run(module, extra, f'C{budget}_K{bits}_{seed}_{module}')
        for module in ['adapted_word_statistics', 'verify_adapted_word_statistics']:
            run(module, ['--budget', str(budget)], f'C{budget}_{module}')
        state['completed_budgets'].append(budget)
        save()
    try:
        while True:
            word_ready = ready('word8_language')
            single_ready = ready('counted_mean_validation')
            if word_ready and single_ready:
                break
            save()
            time.sleep(30)
        evaluate_budget(1024)
        state.update(status='waiting_for_conditional_T4096_completion', current=None)
        save()
        while True:
            current = read(root / 'counted_mean_t4096/pipeline.json')
            if current['status'] == 'extension_not_executed_lower_budget_quality_gate_passed':
                decision = read(root / 'counted_mean_t4096/decision.json')
                lower_stats = root / 'counted_mean_validation/statistics.json'
                lower_audit = root / 'counted_mean_validation/statistics_audit.json'
                assert ready('counted_mean_validation')
                assert not needs_higher_budget(read(lower_stats), read(lower_audit), sha(lower_stats), p['upstream_protocols_sha256']['counted_mean_validation'])
                assert decision['higher_budget_required'] is False
                assert decision['lower_statistics_sha256'] == sha(lower_stats)
                assert decision['lower_statistics_audit_sha256'] == sha(lower_audit)
                state['C4096_status'] = 'not_executed_original_one_pbit_extension_not_needed'
                break
            if ready('counted_mean_t4096'):
                decision = read(root / 'counted_mean_t4096/decision.json')
                lower_stats = root / 'counted_mean_validation/statistics.json'
                lower_audit = root / 'counted_mean_validation/statistics_audit.json'
                assert ready('counted_mean_validation')
                assert needs_higher_budget(read(lower_stats), read(lower_audit), sha(lower_stats), p['upstream_protocols_sha256']['counted_mean_validation'])
                assert decision['higher_budget_required'] is True
                assert decision['lower_statistics_sha256'] == sha(lower_stats)
                assert decision['lower_statistics_audit_sha256'] == sha(lower_audit)
                evaluate_budget(4096)
                state['C4096_status'] = 'all_matched_spatial_cases_and_statistics_verified'
                break
            save()
            time.sleep(30)
        state.update(status='all_eligible_matched_adapted_spatial_cohorts_independently_verified', current=None, child_pid=None)
        save()
    except BaseException as error:
        state.update(status='failed_preserved_no_restart', error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
