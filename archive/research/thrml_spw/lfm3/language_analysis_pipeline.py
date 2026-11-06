"""Wait for the existing cohort, then analyze once; never launch/restart inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_native_v2'
    protocol_path = root / 'language_analysis_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(root / 'language_native_v2_protocol.json') == protocol['cohort_protocol_sha256']
    state = {'status': 'waiting_for_existing_cohort', 'protocol_sha256': sha(protocol_path)}

    def save():
        (out / 'analysis_pipeline.json').write_text(json.dumps(state, indent=2) + '\n')

    while True:
        cohort = json.loads((out / 'pipeline.json').read_text())
        if cohort['status'] == 'all_five_seeds_independently_verified':
            break
        if cohort['status'] != 'running':
            state.update(status='cohort_stopped_preserved_no_restart', cohort_status=cohort['status'])
            save()
            return
        process = Path('/proc') / str(protocol['cohort_pid']) / 'stat'
        if process.exists():
            ticks = process.read_text().rsplit(')', 1)[1].split()[19]
            assert ticks == protocol['cohort_start_ticks'], 'PID identity changed; inspect manually.'
            state.update(last_verified_live_unix=time.time(), completed_seeds=cohort['completed_seeds'], current_seed=cohort['current_seed'])
            save()
        else:
            # Observation alone does not declare inference failed or restart it.
            state.update(status='cohort_process_not_observed_rechecking', last_observation_unix=time.time())
            save()
        time.sleep(30)
    for module in ['language_statistics', 'verify_language_statistics']:
        state.update(status='running_analysis', stage=module)
        save()
        with (out / (module + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module],
                                     stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            code = child.wait()
        if code:
            state.update(status='analysis_failed_preserved', returncode=code)
            save()
            raise RuntimeError(f'{module} failed; outputs preserved')
    state.update(status='five_seed_statistics_independently_verified',
                 statistics_sha256=sha(out / 'statistics.json'),
                 statistics_audit_sha256=sha(out / 'statistics_audit.json'))
    save()


if __name__ == '__main__':
    main()
