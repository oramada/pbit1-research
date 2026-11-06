"""Retrospective BF16 contrasts on two fully audited fixed image cohorts."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path('results/thrml_spw')
OUT = ROOT/'bf16_comparison/evidence'
DATASETS = ['fashion', 'cifar']
CONTROLS = [(1, 'BF16_weights_F64'), (3, 'BF16_Torch_CPU')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, indent=2)+'\n')


def prepare():
    assert not OUT.exists()
    inputs = {}
    for dataset in DATASETS:
        root = ROOT/(dataset+'_cnn_native')
        for folder in ['probability_cohort_evidence', 'precision_evidence']:
            d = root/folder
            completion = json.loads((d/'completion.json').read_text())
            for path, digest in completion['artifact_sha256'].items():
                assert sha(path) == digest, path
                inputs[path] = digest
            for name in ['completion.json', 'pipeline.json', 'protocol.json']:
                inputs[str(d/name)] = sha(d/name)
        for name in ['manifest.json', 'statistics.json', 'statistical_arrays.npz', 'native_audit.json', 'statistics_audit.json']:
            d = root/'probability_cohort_evidence'
            inputs[str(d/name)] = sha(d/name)
        for name in ['statistical_arrays.npz', 'logits.npz', 'evidence_audit.json']:
            d = root/'precision_evidence'
            inputs[str(d/name)] = sha(d/name)
        for name in ['test_data.npz', 'statistical_arrays.npz']:
            d = root/'cohort_evidence'
            inputs[str(d/name)] = sha(d/name)
    sources = {str(path): sha(path) for path in [Path(__file__), Path(__file__).with_name('verify.py')]}
    OUT.mkdir()
    write(OUT/'protocol.json', {
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'datasets': DATASETS, 'methods': ['expected_weight', 'binomial_mean_STE_T256'],
        'samples': [64, 256, 1024], 'BF16_controls': [name for _, name in CONTROLS],
        'all_contrast_count': 24, 'bootstrap_draws': 5000, 'metric_order': ['accuracy_fraction', 'mean_CE_nats'],
        'input_sha256': inputs, 'source_sha256': sources,
        'scope': 'Retrospective descriptive analysis of already consumed, fully audited test images. Both training methods, every budget and both BF16 paths retained. Existing paired model draws and method-specific MC draws reused; no image resampling, new native runs, fresh statistical replicas, equivalence/noninferiority margin/test or multiplicity adjustment. Additional training epochs versus original BF16 baseline remain a confound. BF16 weight rounding with F64 arithmetic and CPU BF16 execution are distinct controls; CPU internal accumulation widths are not asserted.'})
    print('BF16 retrospective analysis frozen before computation: all24 contrasts')


def guard():
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path
    return p


def run():
    p = guard()
    assert not (OUT/'statistics.json').exists()
    contrasts, arrays = [], {}
    for dataset in DATASETS:
        root = ROOT/(dataset+'_cnn_native')
        learned = np.load(root/'probability_cohort_evidence/statistical_arrays.npz')
        precision = np.load(root/'precision_evidence/statistical_arrays.npz')
        base = np.load(root/'cohort_evidence/statistical_arrays.npz')
        values, models, seeds = learned['values'], base['bootstrap_models'], learned['bootstrap_method_seeds']
        assert values.shape == (2, 3, 5, 10, 2) and seeds.shape == (2, 5000, 5, 10)
        for method in range(2):
            for bi, t in enumerate(p['samples']):
                for ci, name in CONTROLS:
                    control = precision['control_metrics'][ci]
                    delta = values[method, bi]-control[:, None, :]
                    boot = delta[models[:, :, None], seeds[method]].mean((1, 2))
                    key = f'{dataset}_method{method}_T{t}_{name}'
                    arrays[key] = boot
                    contrasts.append({'dataset': dataset, 'method_index': method, 'method': p['methods'][method],
                                      'T': t, 'control_index': ci, 'control': name, 'array_key': key,
                                      'native_mean': values[method, bi].mean((0, 1)).tolist(),
                                      'BF16_mean': control.mean(0).tolist(),
                                      'difference': delta.mean((0, 1)).tolist(),
                                      'per_model_difference': delta.mean(1).tolist(),
                                      'descriptive_95_interval': np.quantile(boot, [.025, .975], axis=0).T.tolist()})
    assert len(contrasts) == 24
    np.savez_compressed(OUT/'bootstrap_arrays.npz', **arrays)
    guard()
    write(OUT/'statistics.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'contrasts': contrasts,
                                 'bootstrap_arrays_sha256': sha(OUT/'bootstrap_arrays.npz'), 'scope': p['scope']})
    print('All24 BF16 contrasts computed; independent verification required')


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
