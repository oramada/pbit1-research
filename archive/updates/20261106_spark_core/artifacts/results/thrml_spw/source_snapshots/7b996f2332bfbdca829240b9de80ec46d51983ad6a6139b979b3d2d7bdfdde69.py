"""Isolated synthetic test of paired summaries, gates and negative controls.

No model scores or production result directories are changed by this fixture.
"""
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
    repo = Path.cwd()
    source = Path(__file__).resolve().parent
    names = ['allocated_native_statistics.py', 'verify_allocated_native_statistics.py']
    with tempfile.TemporaryDirectory(prefix='allocated-statistics-') as directory:
        cwd = Path(directory)
        root = cwd / 'results/thrml_spw/lfm3_1p7b'
        root.mkdir(parents=True)
        seeds = list(range(5))
        policy = {f'module{i}': 512 for i in range(287)}
        p = {'seeds': seeds, 'MC_t4_critical': '2.77644510519779866907365875095',
             'BF16_quality_gate_NLL_margin': float(np.log(1.02)), 'mean_program_package_sha256': 'synthetic',
             'sources_sha256': {name: sha(source / name) for name in names},
             'serial_operator_sample_cycles': {'allocated': 287 * 512, 'uniform': 287 * 1024},
             'scope': 'Synthetic fixture only; no LFM3 inference or native quality claim.'}
        (root / 'allocated_native_protocol.json').write_text(json.dumps(p))
        allocdir = root / 'layer_allocation_candidates'; allocdir.mkdir()
        (allocdir / 'results.json').write_text(json.dumps({'selected_samples': policy}))
        def save_result(folder, seed, result):
            case = root / folder / 'mean' / str(seed)
            (case / 'results.json').write_text(json.dumps(result))
            audit = {'status': 'all_direct_program_coverage_losses_and_resource_receipts_verified',
                'results_sha256': sha(case / 'results.json'), 'targets': 2048, 'seed': seed, 'arm': 'mean',
                'losses': {key: {'mean_NLL_FP64': value} for key, value in
                           {'native': (1.005 + seed * .001 if folder == 'allocated_native' else 1.08 + seed * .003),
                            'BF16': 1., 'same_fit_BF16': 1., 'nominal_controls': 1.001}.items()},
                'retained_scheduled_evaluations': 287 * 4 * (512 if folder == 'allocated_native' else 1024),
                'warmup_scheduled_evaluations': 287 * 4}
            (case / 'independent_audit.json').write_text(json.dumps(audit))
        for folder in ['allocated_native', 'adaptation_native_v2']:
            for seed in seeds:
                case = root / folder / 'mean' / str(seed); case.mkdir(parents=True)
                outputs, calls = [], []
                for batch in range(4):
                    path = case / f'batch{batch}.npz'
                    np.savez(path, input_ids=np.array([[batch, 1]]), BF16=np.array([1.]),
                             same_fit_BF16=np.array([1.]), nominal_controls=np.array([1.]))
                    outputs.append({'path': str(path), 'sha256': sha(path)})
                    for i in range(287):
                        calls.append({'batch': batch, 'stage': 'native', 'module': f'module{i}',
                            'canonical_program': f'program{i}', 'program_sha256': str(i),
                            'call': batch * 287 + i, 'weight_coordinates': 1, 'padding_nodes': 0,
                            'warmup_evaluations': 1, 'embedding_rows': None,
                            'retained_evaluations': 512 if folder == 'allocated_native' else 1024})
                result = {'package_sha256': 'synthetic', 'same_fit_BF16_weight_hashes': {'synthetic': 'same'},
                          'outputs': outputs, 'operator_calls': calls}
                save_result(folder, seed, result)
        env = dict(os.environ, PYTHONPATH=str(repo), OPENBLAS_NUM_THREADS='1')
        def run(module, success=True):
            process = subprocess.run([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module],
                cwd=cwd, env=env, text=True, capture_output=True)
            assert (process.returncode == 0) == success, process.stderr
        run('allocated_native_statistics'); run('verify_allocated_native_statistics')
        out = root / 'allocated_native'
        original = (out / 'statistics.json').read_text()
        r = json.loads(original)
        assert r['arms']['allocated']['conditional_BF16_quality_gate_passed']
        assert not r['arms']['uniform']['conditional_BF16_quality_gate_passed']
        assert r['paired_allocated_minus_uniform_NLL']['MC95_t4_interval'][1] < 0
        (out / 'statistics_audit.json').unlink()
        r['paired_allocated_minus_uniform_NLL']['mean'] += .1
        (out / 'statistics.json').write_text(json.dumps(r))
        run('verify_allocated_native_statistics', False)
        (out / 'statistics.json').write_text(original)
        path = root / 'adaptation_native_v2/mean/0/batch0.npz'
        np.savez(path, input_ids=np.array([[0, 1]]), BF16=np.array([2.]),
                 same_fit_BF16=np.array([1.]), nominal_controls=np.array([1.]))
        case = path.parent
        result = json.loads((case / 'results.json').read_text())
        result['outputs'][0]['sha256'] = sha(path)
        save_result('adaptation_native_v2', 0, result)
        (out / 'statistics.json').unlink()
        run('allocated_native_statistics')
        run('verify_allocated_native_statistics', False)
    report = {'status': 'synthetic_paired_arithmetic_both_gate_outcomes_tampered_statistics_and_control_mismatch_verified',
              'source_sha256': sha(Path(__file__)), 'statistics_sources_sha256': p['sources_sha256'],
              'scope': 'Isolated synthetic artifacts only; not full-model inference evidence.'}
    target = repo / 'results/thrml_spw/lfm3_1p7b/allocated_statistics_preflight.json'
    with target.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
