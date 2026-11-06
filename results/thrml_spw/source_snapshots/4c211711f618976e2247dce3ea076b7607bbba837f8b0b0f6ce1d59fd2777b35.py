"""Conditional4096 extension only after all lower-budget seeds and statistics."""
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
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--lower-pid', type=int, required=True); args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b'); shared = root / 'counted_mean_t4096_protocol.json'; p = read(shared)
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items(): assert sha(Path(name)) == digest
    for name, digest in p['inputs_sha256'].items(): assert sha(root / name) == digest
    lower = root / 'counted_mean_validation'; expected = identity(args.lower_pid)
    assert expected and expected['state'] != 'Z' and 'research.thrml_spw.lfm3.counted_mean_validation_pipeline' in expected['argv']
    out = root / 'counted_mean_t4096'; out.mkdir(exist_ok=True)
    with (out / 'pipeline_protocol.json').open('x') as f:
        json.dump({'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(shared),
            'lower_pid': args.lower_pid, 'lower_identity': expected}, f, indent=2)
    state = {'pid': os.getpid(), 'status': 'waiting_for_complete_lower_budget_cohort', 'completed': [], 'protocol_sha256': sha(shared)}
    def save(): (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    while True:
        upstream = read(lower / 'pipeline.json')
        if upstream['status'] == 'all_five_counted_mean_cases_and_statistics_independently_verified': break
        if upstream['status'] == 'failed_preserved_no_restart':
            state.update(status='lower_budget_failed_no_automatic_retry_or_extension', lower_state=upstream); save(); raise SystemExit(1)
        actual = identity(args.lower_pid)
        assert actual and actual['state'] != 'Z' and actual['start_ticks'] == expected['start_ticks'] and actual['argv'] == expected['argv']
        state.update(last_verified_lower_live_unix=time.time(), lower_current=upstream.get('current')); save(); time.sleep(30)
    statistics_path = lower / 'statistics.json'; audit_path = lower / 'statistics_audit.json'
    proceed = needs_higher_budget(read(statistics_path), read(audit_path), sha(statistics_path), p['lower_protocol_sha256'])
    decision = {'higher_budget_required': proceed, 'lower_statistics_sha256': sha(statistics_path),
        'lower_statistics_audit_sha256': sha(audit_path), 'rule': p['continuation_rule'], 'recorded_unix': time.time()}
    with (out / 'decision.json').open('x') as f: json.dump(decision, f, indent=2)
    if not proceed:
        state['status'] = 'extension_not_executed_lower_budget_quality_gate_passed'; save(); return
    state['status'] = 'running'; save()
    def run(module, extra, label):
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
            state.update(current=label, child_pid=child.pid); save(); code = child.wait()
        state['child_exit_code'] = code
        if code:
            state['status'] = 'failed_preserved_no_restart'; save(); raise SystemExit(code)
        state['completed'].append(label); save()
    for seed in p['seeds']:
        for module in ['counted_mean_t4096', 'verify_counted_mean_t4096']:
            run(module, ['--arm', 'mean', '--seed', str(seed)], f'{seed}_{module}')
    for module in ['counted_mean_t4096_statistics', 'verify_counted_mean_t4096_statistics']: run(module, [], module)
    state.update(status='all_five_conditional_T4096_cases_and_statistics_independently_verified', current=None, child_pid=None,
        statistics_audit_sha256=sha(out / 'statistics_audit.json')); save()


if __name__ == '__main__':
    main()
