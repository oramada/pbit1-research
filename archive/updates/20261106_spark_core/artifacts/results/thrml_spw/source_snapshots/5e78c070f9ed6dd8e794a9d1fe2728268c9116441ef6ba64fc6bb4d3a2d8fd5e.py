"""Separate scalar losses and unit-by-unit hierarchical bootstrap verification."""
import datetime
import json
import math
import numpy as np
from results.thrml_spw.fashion_cnn_native.gaussian_control.common import BASE, sha, write, guard
from results.thrml_spw.fashion_cnn_native.gaussian_statistics.study import OUT, PARENT, CONTRASTS


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    data = json.loads((OUT/'statistics.json').read_text())
    assert data['protocol_sha256'] == sha(OUT/'protocol.json') and data['statistical_arrays_sha256'] == sha(OUT/'statistical_arrays.npz')
    manifest = json.loads((PARENT/'manifest.json').read_text())
    assert data['manifest_sha256'] == sha(PARENT/'manifest.json') and data['numerical_audit_sha256'] == sha(PARENT/'numerical_audit.json')
    a = np.load(OUT/'statistical_arrays.npz')
    old = np.load(BASE/'statistical_arrays.npz')
    for current, prior in [('bootstrap_models', 'bootstrap_models'), ('native_bootstrap_seeds', 'bootstrap_seeds')]:
        np.testing.assert_array_equal(a[current], old[prior])
    gs = np.random.default_rng(p['bootstrap_seed']).integers(0, 10, size=(5000, 5, 10))
    np.testing.assert_array_equal(gs, a['Gaussian_bootstrap_seeds'])
    parent = json.loads((PARENT/'protocol.json').read_text())
    labels = np.load(BASE/'test_data.npz')['labels']
    measured = np.empty((2, 3, 5, 10, 2))
    row_lookup = {r['name']: r for r in data['rows']}
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        logits = np.load(row['path'])['logits']
        correct, loss = 0, 0.
        for y, truth in zip(logits, labels):
            correct += max(range(10), key=lambda k: y[k]) == truth
            peak = max(y)
            loss += peak+math.log(sum(math.exp(float(v-peak)) for v in y))-y[int(truth)]
        v = [correct/10000, loss/10000]
        bi = parent['samples'].index(row['T'])
        measured[row['mode'], bi, row['model'], row['seed']] = v
        retained = row_lookup[row['name']]
        np.testing.assert_allclose(v, [retained['accuracy'], retained['mean_CE']], rtol=2e-13, atol=2e-14)
    assert len(row_lookup) == 300
    np.testing.assert_allclose(measured, a['values'], rtol=2e-13, atol=2e-14)
    assert len(data['summaries']) == 6
    for mode in range(2):
        for bi, samples in enumerate(parent['samples']):
            summary = next(r for r in data['summaries'] if r['mode'] == parent['modes'][mode] and r['T'] == samples)
            expected = sum(measured[mode, bi, mi, seed] for mi in range(5) for seed in range(10))/50
            np.testing.assert_allclose(expected, summary['mean'], rtol=2e-13, atol=2e-14)
    checked = 0
    for bi in range(3):
        all_draws = np.zeros((5, 5000, 2))
        for unit in range(5):
            mi = a['bootstrap_models'][:, unit]
            gi = gs[:, unit]
            ni = a['native_bootstrap_seeds'][:, unit]
            g0 = measured[0, bi][mi[:, None], gi].sum(1)/50
            g1 = measured[1, bi][mi[:, None], gi].sum(1)/50
            native = old['values'][bi][mi[:, None], ni].sum(1)/50
            fp32 = old['baseline_metrics'][0][mi]/5
            all_draws += np.array([g0-native, g1-native, g1-g0, g0-fp32, g1-fp32])
        for ci in range(5):
            np.testing.assert_allclose(all_draws[ci], a[f'b{bi}_c{ci}'], rtol=3e-11, atol=2e-13)
            row = next(r for r in data['contrasts'] if r['budget_index'] == bi and r['contrast_index'] == ci)
            assert row['contrast'] == CONTRASTS[ci]
            mode0 = sum(measured[0, bi, mi, seed] for mi in range(5) for seed in range(10))/50
            mode1 = sum(measured[1, bi, mi, seed] for mi in range(5) for seed in range(10))/50
            original = sum(old['values'][bi, mi, seed] for mi in range(5) for seed in range(10))/50
            fp = sum(old['baseline_metrics'][0, mi] for mi in range(5))/5
            point = [mode0-original, mode1-original, mode1-mode0, mode0-fp, mode1-fp][ci]
            np.testing.assert_allclose(point, row['difference'], rtol=3e-11, atol=2e-13)
            np.testing.assert_allclose(np.quantile(all_draws[ci], [.025, .975], axis=0).T, row['descriptive_95_interval'], rtol=3e-11, atol=2e-13)
            checked += all_draws[ci].size
    assert checked == 150000 and len(data['contrasts']) == 15
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'cases_checked': 300, 'scalar_per_example_losses_checked': 3000000,
                                    'bootstrap_scalar_values_checked': checked, 'scope': 'Full scalar metric and unitwise bootstrap checks, descriptive fixed-data comparison only.',
                                    'new_native_updates': 0})
    paths = [OUT/n for n in ['protocol.json', 'statistics.json', 'statistical_arrays.npz', 'evidence_audit.json']]
    paths += [PARENT/n for n in ['protocol.json', 'manifest.json', 'preflight.json', 'numerical_audit.json']]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all300Gaussian_controls_numerically_and_statistically_checked_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All300metrics and150000bootstrap scalar values checked', flush=True)


if __name__ == '__main__':
    main()
