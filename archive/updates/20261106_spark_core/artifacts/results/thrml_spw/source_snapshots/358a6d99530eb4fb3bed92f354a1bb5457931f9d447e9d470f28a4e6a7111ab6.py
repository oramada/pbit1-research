"""Synthetic-only all33learned-RGBcontrast check, with independent scalar draws."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_cohort import run as target
from results.thrml_spw.cifar_cnn_native.review.statistics_fixture import independent_metric

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_statistics_fixture_evidence'
BASE = ROOT/'statistics_fixture_evidence'
WORD = ROOT/'word_statistics_fixture_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def main():
    assert not OUT.exists()
    OUT.mkdir()
    (OUT/'README.md').write_text('# SYNTHETIC ONLY\n\nGenerated20example logits and dummy native success/accounting placeholders exercise statistical code. No task/model/native or physical evidence. Parent closure is substituted only in this fixture process.\n')
    for folder in [BASE, WORD]:
        c = json.loads((folder/'fixture_completion.json').read_text())
        assert c['passed'] and c['fixture_only']
        for path, h in c['artifact_sha256'].items():
            assert sha(path) == h, path
    p = {'source_sha256': {str(Path(target.__file__)): sha(target.__file__), str(Path(__file__)): sha(__file__)},
         'input_sha256': {}, 'samples': [64, 256, 1024], 'bootstrap_seed': 1877520,
         'statistics': 'SYNTHETIC ONLY: all33contrast arithmetic, no real task/native evidence.'}
    write(OUT/'protocol.json', p)
    labels = np.load(BASE/'test_data.npz')['labels']
    rng = np.random.default_rng(1887520)
    values = np.empty((2, 3, 5, 10, 2))
    means = np.empty((2, 5, 2))
    plugin, rows = {}, []
    for method in range(2):
        for mi in range(5):
            z = rng.normal(size=(20, 10))
            z[np.arange(20), labels] += .3*method-.07*mi
            plugin[f'method{method}_model{mi}'] = z
            means[method, mi] = independent_metric(z, labels)
            for bi, t in enumerate(p['samples']):
                for seed in range(10):
                    z = rng.normal(size=(20, 10))*(.7+.03*seed)
                    z[np.arange(20), labels] += .2*method+.06*mi-.11*bi
                    values[method, bi, mi, seed] = independent_metric(z, labels)
                    name = f'method{method}_model{mi}_T{t}_seed{seed}'
                    path = OUT/(name+'.npz')
                    np.savez_compressed(path, logits=z)
                    rows.append({'method': method, 'model': mi, 'T': t, 'seed': seed,
                                 'name': name, 'path': str(path), 'sha256': sha(path), 'clipped_coordinates': [0, 0, 0]})
    np.savez_compressed(OUT/'mean_logits.npz', **plugin)
    write(OUT/'manifest.json', {'fixture_only': True, 'complete': True, 'rows': rows})
    audit_rows = [{'name': r['name'], 'case_sha256': r['sha256'], 'all17500_array_receipts_exact': True,
                   'all100000_final_logits_exact': True} for r in rows]
    write(OUT/'native_audit.json', {'fixture_only': True, 'passed': True, 'rows': audit_rows,
          'manifest_sha256': sha(OUT/'manifest.json'), 'counter_entries_exact': 18474750000, 'public_updates': 659904000000})
    original = target.OUT, target.BASE, target.WORD, target.parent_closed
    target.OUT, target.BASE, target.WORD, target.parent_closed = OUT, BASE, WORD, lambda: None
    try:
        target.statistics()
    finally:
        target.OUT, target.BASE, target.WORD, target.parent_closed = original
    base, word = np.load(BASE/'statistical_arrays.npz'), np.load(WORD/'statistical_arrays.npz')
    arrays = np.load(OUT/'statistical_arrays.npz')
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(means, arrays['mean_metrics'], rtol=2e-13, atol=2e-14)
    bs = np.random.default_rng(1877520).integers(0, 10, (2, 5000, 5, 10))
    np.testing.assert_array_equal(bs, arrays['bootstrap_method_seeds'])
    bm, seen = base['bootstrap_models'], set()
    for c in json.loads((OUT/'statistics.json').read_text())['contrasts']:
        bi, name = c['budget_index'], c['contrast']
        assert (bi, name) not in seen
        seen.add((bi, name))
        method = int(name.startswith('sampled'))
        suffix = name.split('_minus_')[1]
        if suffix == 'original':
            right, indices = base['values'][bi], base['bootstrap_seeds']
        elif suffix == 'expected':
            right, indices = values[0, bi], bs[0]
        elif suffix == 'word':
            right, indices = word['values'][bi], word['bootstrap_word_seeds']
        else:
            assert suffix in ['own_mean', 'FP32', 'INT4']
            right = means[method] if suffix == 'own_mean' else base['baseline_metrics'][0 if suffix == 'FP32' else 4]
            indices = None
        reference = np.empty((5000, 2))
        for draw in range(5000):
            for metric in range(2):
                lhs = math.fsum(float(values[method, bi, bm[draw, unit], bs[method, draw, unit, sample], metric]) for unit in range(5) for sample in range(10))/50
                rhs = math.fsum(float(right[bm[draw, unit], metric]) for unit in range(5))/5 if indices is None else math.fsum(float(right[bm[draw, unit], indices[draw, unit, sample], metric]) for unit in range(5) for sample in range(10))/50
                reference[draw, metric] = lhs-rhs
        np.testing.assert_allclose(reference, arrays[f'{name}_budget{bi}'], rtol=2e-9, atol=3e-12)
        point = values[method, bi].mean((0, 1))-(right.mean(0) if indices is None else right.mean((0, 1)))
        np.testing.assert_allclose(point, c['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(reference, [.025, .975], axis=0).T, c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert seen == {(bi, name) for bi in range(3) for name in target.CONTRASTS}
    write(OUT/'fixture_completion.json', {'fixture_only': True, 'passed': True,
          'case_metrics_checked': 300, 'control_metrics_checked': 10,
          'independently_enumerated_bootstrap_scalars': 330000, 'native_calls': 0,
          'artifact_sha256': {str(path): sha(path) for path in OUT.iterdir() if path.is_file()}})
    destination = ROOT/'probability_cohort_evidence/statistics_selfcheck.json'
    assert not destination.exists()
    write(destination, {'passed': True, 'fixture_only': True, 'contrasts': 33,
          'independently_enumerated_bootstrap_scalars': 330000, 'source_sha256': sha(__file__),
          'fixture_completion_sha256': sha(OUT/'fixture_completion.json')})
    print('SYNTHETIC ONLY: all33learnedRGBcontrasts/330000bootstrap scalar values match', flush=True)


if __name__ == '__main__':
    main()
