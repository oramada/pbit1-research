"""Separate scalar metrics, model-unit bootstrap checks and prospective fixture."""
import datetime
import json
import math
import sys
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard
from results.thrml_spw.fashion_cnn_native.allocated_statistics.study import OUT, BASE, PARENT, NAMES, draws


def unit_draws(values, baseline, bm, bs):
    results = np.zeros((3, len(bm), 2))
    for unit in range(5):
        models = bm[:, unit]
        ga = values[0][models[:, None], bs[0, :, unit]].sum(1)/50
        gb = values[1][models[:, None], bs[1, :, unit]].sum(1)/50
        f = baseline[models]/5
        results += np.stack([gb-ga, ga-f, gb-f])
    return results


def selfcheck():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'selfcheck.json').exists()
    values = np.empty((2, 5, 10, 2))
    for ci in range(2):
        for mi in range(5):
            for seed in range(10):
                values[ci, mi, seed] = [.5+.01*ci+.003*mi+.0001*seed*(mi+1+ci), 1+.2*ci+.01*mi+.001*seed*(mi+1+2*ci)]
    baseline = np.array([[.7+.001*mi, .8+.002*mi] for mi in range(5)])
    rng = np.random.default_rng(p['bootstrap_seed'])
    bm = rng.integers(0, 5, size=(5000, 5))
    bs = rng.integers(0, 10, size=(2, 5000, 5, 10))
    assert not np.array_equal(bs[0], bs[1])
    actual = draws(values, baseline, bm, bs)
    expected = unit_draws(values, baseline, bm, bs)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
    # A fully scalar calculation of the first100draws also checks the unit checker.
    scalar = np.empty((3, 100, 2))
    for d in range(100):
        a, b, f = np.zeros(2), np.zeros(2), np.zeros(2)
        for unit in range(5):
            mi = int(bm[d, unit])
            f += baseline[mi]/5
            for j in range(10):
                a += values[0, mi, bs[0, d, unit, j]]/50
                b += values[1, mi, bs[1, d, unit, j]]/50
        scalar[:, d] = np.stack([b-a, a-f, b-f])
    np.testing.assert_allclose(actual[:, :100], scalar, rtol=3e-12, atol=3e-14)
    write(OUT/'selfcheck.json', {'passed': True, 'protocol_sha256': sha(OUT/'protocol.json'),
                               'fixture_bootstrap_values_checked': 30000, 'fully_scalar_values_checked': 600,
                               'scope': 'Synthetic model/seed-interaction fixture checks vectorized versus separateunitaccumulation and scalar double loops, independent percondition seed arrays. No real model, labels, task scores or native calls.'})
    print('Prospective bootstrap fixture passes30000values,600fullyscalar values', flush=True)


def scalar_metrics(logits, labels):
    correct, loss = 0, 0.
    for y, truth in zip(logits, labels):
        correct += max(range(10), key=lambda k: y[k]) == truth
        peak = max(y)
        loss += peak+math.log(sum(math.exp(float(v-peak)) for v in y))-y[int(truth)]
    return np.array([correct/len(labels), loss/len(labels)])


def verify():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert json.loads((OUT/'selfcheck.json').read_text())['passed']
    d = json.loads((OUT/'statistics.json').read_text())
    assert d['protocol_sha256'] == sha(OUT/'protocol.json') and d['statistical_arrays_sha256'] == sha(OUT/'statistical_arrays.npz')
    assert d['manifest_sha256'] == sha(PARENT/'manifest.json') and d['native_audit_sha256'] == sha(PARENT/'native_audit.json')
    parent = json.loads((PARENT/'protocol.json').read_text())
    guard(parent)
    m = json.loads((PARENT/'manifest.json').read_text())
    labels = np.load(BASE/'test_data.npz')['labels']
    a = np.load(OUT/'statistical_arrays.npz')
    controls = np.load(BASE/'control_logits.npz')
    baseline = np.stack([scalar_metrics(controls[f'c0_model{mi}'], labels) for mi in range(5)])
    np.testing.assert_allclose(baseline, a['baseline'], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(baseline.mean(0), d['baseline_mean'], rtol=2e-13, atol=2e-14)
    values = np.empty((2, 5, 10, 2))
    lookup = {r['name']: r for r in d['rows']}
    assert len(lookup) == len(m['rows']) == 100
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        case = np.load(row['path'])
        v = scalar_metrics(case['logits'], labels)
        values[row['condition'], row['model'], row['seed']] = v
        retained = lookup[row['name']]
        assert [retained[k] for k in ['condition', 'model', 'seed', 'budgets']] == [row[k] for k in ['condition', 'model', 'seed', 'budgets']]
        np.testing.assert_array_equal(case['clipped_coordinates'], retained['clipped_coordinates'])
        np.testing.assert_allclose(v, [retained['accuracy'], retained['mean_CE']], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(values, a['values'], rtol=2e-13, atol=2e-14)
    rng = np.random.default_rng(p['bootstrap_seed'])
    bm = rng.integers(0, 5, size=(5000, 5))
    bs = rng.integers(0, 10, size=(2, 5000, 5, 10))
    np.testing.assert_array_equal(bm, a['bootstrap_models'])
    np.testing.assert_array_equal(bs, a['bootstrap_seeds'])
    independent = unit_draws(values, baseline, bm, bs)
    np.testing.assert_allclose(independent, a['distributions'], rtol=3e-11, atol=2e-13)
    means = [sum(values[ci, mi, seed] for mi in range(5) for seed in range(10))/50 for ci in range(2)]
    f = sum(baseline)/5
    points = [means[1]-means[0], means[0]-f, means[1]-f]
    assert len(d['contrasts']) == 3 and len(d['summaries']) == 2
    for ci in range(2):
        summary = d['summaries'][ci]
        assert summary['condition'] == parent['conditions'][ci]
        np.testing.assert_allclose(summary['mean'], means[ci], rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(summary['model_means'], values[ci].mean(1), rtol=2e-13, atol=2e-14)
    for i, row in enumerate(d['contrasts']):
        assert row['contrast'] == NAMES[i]
        np.testing.assert_allclose(points[i], row['difference'], rtol=3e-11, atol=2e-13)
        np.testing.assert_allclose(np.quantile(independent[i], [.025, .975], axis=0).T, row['descriptive_95_interval'], rtol=3e-11, atol=2e-13)
    resource = json.loads((OUT/'resource_ledger.json').read_text())
    original = json.loads((BASE/'resource_ledger.json').read_text())
    assert resource['base_parameter_payload_bytes'] == original['parameter_payload_bytes'] == 3392
    assert resource['logical_payload_with_schedule_bytes'] == {'uniform': 3394, 'allocated': 3398}
    assert resource['model_layer_samples'] == parent['schedules']
    for samples in resource['model_layer_samples']:
        assert sum(n*t for n, t in zip([72, 1152, 1440], samples)) == 681984
    assert resource['full_production_retained_sample_updates'] == resource['full_public_retained_sample_updates'] == 85248000000
    assert resource['full_warmup_updates_per_pass'] == 1250*100*2664 == 333000000
    assert resource['bank_counter_entries_int32']*4 == resource['bank_counter_bytes'] == 150984
    preflight = json.loads((PARENT/'preflight.json').read_text())
    for field in ['native', 'public']:
        assert resource[f'preflight_{field}_retained_updates'] == preflight[f'{field}_updates']
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'new_cases_checked': 100, 'per_example_scalar_losses_checked': 1050000,
                                    'bootstrap_scalar_values_checked': 30000, 'all_summaries_contrasts_intervals_and_resource_identities_checked': True,
                                    'scope': 'Scalar metrics and separateunitbootstrap checks of the fixed consumed-data matched-update study. No extra native sampling, fresh generalization or physical advantage claim.'})
    paths = [OUT/n for n in ['protocol.json', 'selfcheck.json', 'statistics.json', 'statistical_arrays.npz', 'resource_ledger.json', 'evidence_audit.json']]
    paths += [PARENT/n for n in ['protocol.json', 'manifest.json', 'native_audit.json', 'preflight.json']]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all100matched_budget_CNN_cases_independently_checked_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All100cases,1050000scalar losses and30000bootstrap values verified', flush=True)


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'verify': verify}[sys.argv[1]]()
