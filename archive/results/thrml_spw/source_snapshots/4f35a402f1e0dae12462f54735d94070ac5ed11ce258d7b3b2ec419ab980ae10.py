"""All 24 paired-model, independent-MC matched-work descriptive contrasts."""
import json
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.learned_word_cohort.common import (
    OUT, BASE, SAME, METHODS, CONTRASTS, sha, write, save, guard, closed, metrics, contrast, singles)


def selfcheck():
    assert not (OUT/'statistics_selfcheck.json').exists()
    rng = np.random.default_rng(1947520)
    left = rng.normal(size=(5, 10, 2)); other = rng.normal(size=(5, 10, 2)); det = rng.normal(size=(5, 2))
    bm = rng.integers(0, 5, (7, 5)); ls = rng.integers(0, 10, (7, 5, 10)); rs = rng.integers(0, 10, (7, 5, 10))
    for right, seeds in [(other, rs), (det, None)]:
        point, distribution = contrast(left, right, bm, ls, seeds)
        expected = np.zeros((7, 2))
        for draw in range(7):
            for unit in range(5):
                model = bm[draw, unit]
                for replica in range(10):
                    expected[draw] += left[model, ls[draw, unit, replica]]/50
                    expected[draw] -= (right[model] if seeds is None else right[model, seeds[draw, unit, replica]])/50
        np.testing.assert_allclose(distribution, expected, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(point, left.mean((0, 1))-(right.mean(0) if seeds is None else right.mean((0, 1))))
    write(OUT/'statistics_selfcheck.json', {'passed': True, 'scalar_values_checked': 28,
        'source_sha256': sha(__file__), 'scope': 'Synthetic independent-MC/deterministic bootstrap indexing; no task evidence.'})


def run():
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    assert not (OUT/'statistics.json').exists()
    manifest = json.loads((OUT/'manifest.json').read_text()); audit = json.loads((OUT/'native_audit.json').read_text())
    assert manifest['complete'] and audit['passed'] and audit['complete']
    assert audit['manifest_sha256'] == sha(OUT/'manifest.json')
    assert len(manifest['rows']) == len(audit['rows']) == 400
    assert audit['scheduled_updates'] == manifest['scheduled_updates'] == p['retained_updates_per_pass']
    assert audit['counter_entries'] == p['counter_entries']
    proofs = {r['name']: r for r in audit['rows']}; assert len(proofs) == 400
    y = np.load(BASE/'test_data.npz')['labels']; values = np.empty((2, 4, 5, 10, 2)); rows = []
    for row in manifest['rows']:
        assert proofs[row['name']]['all_receipts_and_final_logits_exact'] and proofs[row['name']]['sha256'] == row['sha256']
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as data:
            value = metrics(data['logits'], y)
        bi = p['word_samples'].index(row['T']); values[row['method'], bi, row['model'], row['seed']] = value
        rows.append({**{k: row[k] for k in ['name', 'method', 'model', 'T', 'seed']},
            'accuracy': float(value[0]), 'mean_CE': float(value[1])})
    with np.load(OUT/'mean_logits.npz') as z:
        means = np.array([[metrics(z[f'method{method}_model{mi}'], y) for mi in range(5)] for method in range(2)])
    single, single_seeds = singles(); bf16 = np.load(SAME/'statistical_arrays.npz')['control_metrics'][:, 1]
    bm = np.load(BASE/'statistical_arrays.npz')['bootstrap_models']
    word_seeds = np.random.default_rng(p['bootstrap_word_seed']).integers(0, 10, (2, 5000, 5, 10))
    distributions = {}; contrasts = []; summaries = []
    for method in range(2):
        for bi, t in enumerate(p['word_samples']):
            word = values[method, bi]
            summaries.append({'method_index': method, 'method': METHODS[method], 'T_word': t, 'T_single': 4*t,
                'mean': word.mean((0, 1)).tolist(), 'model_means': word.mean(1).tolist(),
                'clipped_coordinates': np.sum([r['clipped_coordinates'] for r in manifest['rows']
                    if r['method'] == method and r['T'] == t], axis=0).tolist()})
            for name in CONTRASTS:
                if name == 'single_minus_word':
                    point, dist = contrast(single[method, bi], word, bm, single_seeds[bi][method], word_seeds[method])
                else:
                    right = means[method] if name == 'word_minus_nominal_word_mean' else bf16[method]
                    point, dist = contrast(word, right, bm, word_seeds[method])
                key = f'method{method}_budget{bi}_{name}'; distributions[key] = dist
                contrasts.append({'method_index': method, 'method': METHODS[method], 'budget_index': bi,
                    'T_word': t, 'T_single': 4*t, 'contrast': name, 'array_key': key,
                    'difference': point.tolist(), 'descriptive_95_interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz', values=values, mean_metrics=means, bootstrap_word_seeds=word_seeds, **distributions)
    write(OUT/'statistics.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'rows': rows,
        'summaries': summaries, 'contrasts': contrasts, 'mean_control_summary': means.mean(1).tolist(),
        'artifact_sha256': {str(OUT/name): sha(OUT/name) for name in ['manifest.json', 'native_audit.json',
            'mean_logits.npz', 'mean_weights.npz', 'statistical_arrays.npz']},
        'metric_order': ['accuracy_fraction', 'mean_CE_nats'], 'scope': p['statistics']})


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'run': run}[sys.argv[1]]()
