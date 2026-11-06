"""Isolated synthetic checks of mean-quality statistics and rejection paths."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path.cwd(); source = Path(__file__).resolve().parent
    names = ['counted_mean_statistics.py', 'verify_counted_mean_statistics.py', 'adaptation_native_v2_statistics.py']
    with tempfile.TemporaryDirectory(prefix='counted-mean-statistics-') as temporary:
        cwd = Path(temporary); root = cwd / 'results/thrml_spw/lfm3_1p7b'; root.mkdir(parents=True)
        selection = root / 'adaptation_native_v2/surrogate_selection.json'; selection.parent.mkdir()
        selection.write_text(json.dumps({'selected_arm': 'mean'}))
        p = {'seeds': list(range(5)), 'MC_t4_critical': '2.77644510519779866907365875095',
             'BF16_quality_gate_NLL_margin': float(np.log(1.02)), 'selection_sha256': sha(selection),
             'sources_sha256': {n: sha(source / n) for n in names}, 'scope': 'Synthetic fixture only.'}
        (root / 'counted_mean_validation_protocol.json').write_text(json.dumps(p))
        out = root / 'counted_mean_validation'
        def fixtures(base):
            for seed in p['seeds']:
                case = out / 'mean' / str(seed); case.mkdir(parents=True, exist_ok=True)
                (case / 'results.json').write_text(json.dumps({'synthetic': True, 'seed': seed}))
                audit = {'status': 'all_direct_program_coverage_losses_and_resource_receipts_verified',
                    'seed': seed, 'arm': 'mean', 'targets': 2048, 'unique_parameter_coordinates': 1728801536,
                    'results_sha256': sha(case / 'results.json'),
                    'losses': {k: {'mean_NLL_FP64': v} for k, v in {'native': base + seed * .001,
                        'BF16': 1.1, 'same_fit_BF16': 1., 'nominal_controls': 1.001}.items()},
                    'retained_scheduled_evaluations': 1024, 'warmup_scheduled_evaluations': 1,
                    'max_temporary_weight_count_bytes': 4}
                (case / 'independent_audit.json').write_text(json.dumps(audit))
        env = dict(os.environ, PYTHONPATH=str(repo), OPENBLAS_NUM_THREADS='1')
        def run(module, success=True):
            result = subprocess.run([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module],
                                    cwd=cwd, env=env, capture_output=True, text=True)
            assert (result.returncode == 0) == success, result.stderr
        fixtures(1.005)
        run('counted_mean_statistics'); run('verify_counted_mean_statistics')
        stats = out / 'statistics.json'; original = json.loads(stats.read_text())
        assert original['conditional_same_fit_BF16_gate_passed']
        (out / 'statistics_audit.json').unlink()
        original['conditional_same_fit_BF16_gate_passed'] = False
        stats.write_text(json.dumps(original)); run('verify_counted_mean_statistics', False)
        stats.unlink(); fixtures(1.2)
        run('counted_mean_statistics'); run('verify_counted_mean_statistics')
        original = json.loads(stats.read_text()); assert not original['conditional_same_fit_BF16_gate_passed']
        (out / 'statistics_audit.json').unlink()
        original['native_minus_control_NLL']['same_fit_BF16']['MC95_t4_interval'][1] -= .1
        stats.write_text(json.dumps(original)); run('verify_counted_mean_statistics', False)
    report = {'status': 'synthetic_five_seed_statistics_both_gate_outcomes_and_tamper_rejections_verified',
        'source_sha256': sha(Path(__file__)), 'statistics_sources_sha256': p['sources_sha256'],
        'scope': 'Synthetic arithmetic fixtures only, no model outcomes or native quality; isolated temporary directories.'}
    with (repo / 'results/thrml_spw/lfm3_1p7b/counted_mean_statistics_preflight.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
