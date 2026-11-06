"""Synthetic-only adversarial check of paired-model/independent-MC statistics."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.word_cohort import statistics as target
from results.thrml_spw.cifar_cnn_native.review.statistics_fixture import independent_metric

OUT = Path('results/thrml_spw/cifar_cnn_native/word_statistics_fixture_evidence')
PARENT = Path('results/thrml_spw/cifar_cnn_native/statistics_fixture_evidence')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def main():
    assert not OUT.exists()
    parent_completion = json.loads((PARENT/'fixture_completion.json').read_text())
    assert parent_completion['fixture_only'] and parent_completion['passed']
    for path, h in parent_completion['artifact_sha256'].items():
        assert sha(path) == h, path
    OUT.mkdir()
    (OUT/'README.md').write_text('# SYNTHETIC ONLY\n\nGenerated logits and explicit dummy native success/accounting placeholders exercise statistical code. No model task quality, native replay or physical evidence.\n')
    contrasts = ['word_minus_single', 'word_minus_FP32', 'word_minus_INT4_minmax', 'word_minus_nominal_word_mean']
    p = {'source_sha256': {str(Path(target.__file__)): sha(target.__file__), str(Path(__file__)): sha(__file__)},
         'input_sha256': {}, 'fixture_only': True, 'word_samples': [16, 64, 256],
         'contrasts': contrasts, 'bootstrap_word_seed': 1787520, 'statistics': 'SYNTHETIC FIXTURE ONLY'}
    write(OUT/'protocol.json', p)
    rng = np.random.default_rng(1797520)
    labels = np.load(PARENT/'test_data.npz')['labels']
    means = np.empty((5, 2))
    plugins = {}
    for mi in range(5):
        z = rng.normal(size=(20, 10))
        z[np.arange(20), labels] += .2*mi
        plugins[f'model{mi}'] = z
        means[mi] = independent_metric(z, labels)
    np.savez_compressed(OUT/'mean_logits.npz', **plugins)
    np.savez_compressed(OUT/'mean_weights.npz', fixture_placeholder=np.array([True]))
    values = np.empty((3, 5, 10, 2))
    rows = []
    for mi in range(5):
        for bi, t in enumerate(p['word_samples']):
            for seed in range(10):
                z = rng.normal(size=(20, 10))*(.8+.05*seed)
                z[np.arange(20), labels] += .1*bi-.15*mi+.03*seed
                values[bi, mi, seed] = independent_metric(z, labels)
                name = f'word_model{mi}_T{t}_seed{seed}'
                path = OUT/(name+'.npz')
                np.savez_compressed(path, logits=z)
                rows.append({'name': name, 'model': mi, 'T': t, 'seed': seed,
                             'path': str(path), 'sha256': sha(path), 'clipped_coordinates': [0, 0, 0],
                             'all_receipts_and_final_logits_exact': True})
    write(OUT/'manifest.json', {'fixture_only': True, 'complete': True, 'rows': rows})
    write(OUT/'native_audit.json', {'fixture_only': True, 'passed': True, 'complete': True,
          'manifest_sha256': sha(OUT/'manifest.json'), 'rows': rows,
          'counter_entries': 9237375000, 'scheduled_updates': 329952000000})
    original = target.OUT, target.PARENT, target.parent_closed
    target.OUT, target.PARENT, target.parent_closed = OUT, PARENT, lambda: None
    try:
        target.main()
    finally:
        target.OUT, target.PARENT, target.parent_closed = original
    parent = np.load(PARENT/'statistical_arrays.npz')
    arrays = np.load(OUT/'statistical_arrays.npz')
    bm, bs = parent['bootstrap_models'], parent['bootstrap_seeds']
    bw = np.random.default_rng(p['bootstrap_word_seed']).integers(0, 10, (5000, 5, 10))
    np.testing.assert_array_equal(bw, arrays['bootstrap_word_seeds'])
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(means, arrays['mean_metrics'], rtol=2e-13, atol=2e-14)
    seen = set()
    for c in json.loads((OUT/'statistics.json').read_text())['contrasts']:
        bi, name = c['budget_index'], c['contrast']
        assert (bi, name) not in seen
        seen.add((bi, name))
        reference = parent['values'][bi] if name == 'word_minus_single' else means if name == 'word_minus_nominal_word_mean' else parent['baseline_metrics'][0 if name == 'word_minus_FP32' else 4]
        refmean = reference.mean((0, 1)) if name == 'word_minus_single' else reference.mean(0)
        independent = np.empty((5000, 2))
        for draw in range(5000):
            for metric in range(2):
                word = math.fsum(float(values[bi, bm[draw, unit], bw[draw, unit, sample], metric]) for unit in range(5) for sample in range(10))/50
                if name == 'word_minus_single':
                    ref = math.fsum(float(reference[bm[draw, unit], bs[draw, unit, sample], metric]) for unit in range(5) for sample in range(10))/50
                else:
                    ref = math.fsum(float(reference[bm[draw, unit], metric]) for unit in range(5))/5
                independent[draw, metric] = word-ref
        np.testing.assert_allclose(independent, arrays[f'{name}_budget{bi}'], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(c['difference'], values[bi].mean((0, 1))-refmean, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(c['descriptive_95_interval'], np.quantile(independent, [.025, .975], axis=0).T, rtol=2e-10, atol=2e-12)
    assert seen == {(bi, name) for bi in range(3) for name in contrasts}
    write(OUT/'fixture_completion.json', {'fixture_only': True, 'passed': True,
          'independently_enumerated_bootstrap_scalars': 120000, 'native_calls': 0,
          'scope': 'Output redirection and parent closure substitution are confined to this synthetic-only process; actual source/hash guard and statistics execute unchanged.',
          'artifact_sha256': {str(path): sha(path) for path in OUT.iterdir() if path.is_file()}})
    print('SYNTHETIC ONLY: all12wordcontrasts/120000bootstrap values independently enumerated', flush=True)


if __name__ == '__main__':
    main()
