"""New uniform8192 condition after both completed lower budgets missed the gate.

Fresh five-key cohort, no retries or additional-seed stopping at an old budget.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .adaptation_native_v2_pipeline import read
from .count_budget_continuation import needs_higher_budget


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared = root / 'counted_mean_t8192_protocol.json'
    p = read(shared)
    assert p['samples'] == 8192 and len(p['seeds']) == len(set(p['seeds'])) == 5
    assert 2047 * 128 * p['samples'] == 2146435072 < 2 ** 31 - 1
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in p['inputs_sha256'].items():
        assert sha(root / name) == digest
    parents = {}
    for name in ['counted_mean_validation', 'counted_mean_t4096']:
        folder = root / name
        statistics, audit = read(folder / 'statistics.json'), read(folder / 'statistics_audit.json')
        assert audit['source_sha256'] == p['parent_statistics_verifier_sha256'][name]
        assert needs_higher_budget(statistics, audit, sha(folder / 'statistics.json'), p['parent_protocols_sha256'][name])
        parents[name] = {'statistics_sha256': sha(folder / 'statistics.json'),
                         'statistics_audit_sha256': sha(folder / 'statistics_audit.json')}
    out = root / 'counted_mean_t8192'
    out.mkdir(exist_ok=False)
    with (out / 'pipeline_protocol.json').open('x') as handle:
        json.dump({'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(shared),
                   'parent_failure_receipts': parents, 'verified_int32_bound': 2146435072}, handle, indent=2)
    state = {'pid': os.getpid(), 'status': 'preparing_new_8192_condition', 'completed': [], 'protocol_sha256': sha(shared)}
    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    def resources():
        while True:
            available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            fs = os.statvfs(root)
            free_disk = fs.f_bavail * fs.f_frsize
            state.update(available_host_kB=available, free_disk_bytes=free_disk)
            if available >= p['minimum_available_host_kB'] and free_disk >= p['minimum_free_disk_bytes']:
                return
            state.update(status='waiting_for_memory_or_disk_headroom', child_pid=None)
            save()
            time.sleep(30)
    def run(module, extra, label):
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module, *extra], stdout=log, stderr=log)
            state.update(status='running', current=label, child_pid=child.pid)
            save()
            code = child.wait()
        state['child_exit_code'] = code
        assert code == 0, f'{label} exited {code}; no retry'
        state['completed'].append(label)
        state.update(child_pid=None)
        save()
    save()
    try:
        for seed in p['seeds']:
            resources()
            for module in ['counted_mean_t8192', 'verify_counted_mean_t8192']:
                run(module, ['--arm', 'mean', '--seed', str(seed)], f'{seed}_{module}')
        for module in ['counted_mean_t8192_statistics', 'verify_counted_mean_t8192_statistics']:
            run(module, [], module)
        state.update(status='all_five_T8192_cases_and_statistics_independently_verified', current=None, child_pid=None,
                     statistics_audit_sha256=sha(out / 'statistics_audit.json'))
        save()
    except BaseException as error:
        state.update(status='failed_preserved_no_restart', error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
