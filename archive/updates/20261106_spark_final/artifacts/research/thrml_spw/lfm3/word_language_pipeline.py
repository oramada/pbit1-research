"""Queue matched four-node cohort after the existing single-node cohort/audits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol_path = root / 'word_language_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(root / 'word_preflight/independent_audit.json') == protocol['preflight_audit_sha256']
    out = root / 'word_language'
    out.mkdir(exist_ok=True)
    state = {'protocol_sha256': sha(protocol_path), 'pid': os.getpid(),
             'completed_seeds': [], 'status': 'waiting_for_single_cohort_and_analysis'}

    def save():
        (out / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n')

    while True:
        single = json.loads((root / 'language_native_v2/pipeline.json').read_text())
        analysis = json.loads((root / 'language_native_v2/analysis_pipeline.json').read_text())
        if analysis['status'] == 'five_seed_statistics_independently_verified':
            assert single['status'] == 'all_five_seeds_independently_verified'
            break
        if single['status'] not in ['running', 'all_five_seeds_independently_verified'] or analysis['status'] in ['analysis_failed_preserved', 'cohort_stopped_preserved_no_restart']:
            state.update(status='upstream_stopped_preserved_no_restart', upstream_single=single['status'], upstream_analysis=analysis['status'])
            save()
            return
        proc = Path('/proc') / str(protocol['upstream_cohort_pid']) / 'stat'
        if single['status'] == 'running' and proc.exists():
            assert proc.read_text().rsplit(')', 1)[1].split()[19] == protocol['upstream_cohort_start_ticks']
            state['last_verified_upstream_live_unix'] = time.time()
        state.update(upstream_completed_seeds=single['completed_seeds'], last_observation_unix=time.time())
        save()
        time.sleep(30)
    assert sha(root / 'language_native_v2_protocol.json') == protocol['single_cohort_protocol_sha256']
    for seed in protocol['seeds']:
        assert not (out / str(seed) / 'protocol.json').exists(), 'Existing seed; no automatic restart.'
        state.update(status='running', current_seed=seed)
        for module, suffix in [('word_language', 'native'), ('verify_word_language', 'audit')]:
            state['stage'] = suffix
            save()
            with (out / f'{seed}_{suffix}.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module,
                                          '--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save()
                code = child.wait()
            if code:
                state.update(status='failed_preserved_no_restart', returncode=code)
                save()
                raise RuntimeError(f'{module} seed{seed} failed; preserve logs')
        audit = json.loads((out / str(seed) / 'independent_audit.json').read_text())
        assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
        assert audit['retained_work_matches_paired_single_T1024'] and audit['warmup_four_times_paired_single']
        state['completed_seeds'].append(seed)
        save()
    state.update(status='all_five_word_seeds_independently_verified', stage='complete')
    save()


if __name__ == '__main__':
    main()
