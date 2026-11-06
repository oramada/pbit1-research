"""Full-cohort statistics, gated on complete public replay; no partial scores."""
import json
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_high_budget.common import (
    ROOT, OUT, PARENT, BASE, WORD, PRECISION, METHODS, CONTRASTS,
    sha, write, save, guard, parent_closed, metrics, references, compute)


def selfcheck():
    assert not (OUT/'statistics_selfcheck.json').exists()
    rng = np.random.default_rng(1897519)
    values = rng.normal(size=(2, 5, 10, 2))
    refs = {'BF16_weights_F64': rng.normal(size=(5, 2)), 'BF16_Torch_CPU': rng.normal(size=(5, 2)),
            'FP32_F64': rng.normal(size=(5, 2)), 'own_mean': rng.normal(size=(2, 5, 2)),
            'own_T1024': rng.normal(size=(2, 5, 10, 2)), 'original_T1024': rng.normal(size=(5, 10, 2)),
            'word_T256': rng.normal(size=(5, 10, 2))}
    bm = rng.integers(0, 5, (7, 5))
    bs = rng.integers(0, 10, (2, 7, 5, 10))
    ps = rng.integers(0, 10, (2, 7, 5, 10))
    bas = rng.integers(0, 10, (7, 5, 10))
    ws = rng.integers(0, 10, (7, 5, 10))
    rows = compute(values, refs, bm, bs, ps, bas, ws)
    assert len(rows) == 15
    for row in rows:
        method, name = row['method'], row['reference']
        target = values[0] if name == 'expected_T4096' else refs[name]
        if name in ['own_mean', 'own_T1024']:
            target = target[method]
        seed = ps[method] if name == 'own_T1024' else (bas if name == 'original_T1024' else (ws if name == 'word_T256' else bs[0]))
        expected = np.zeros((7, 2))
        for draw in range(7):
            for unit in range(5):
                mi = int(bm[draw, unit])
                for j in range(10):
                    expected[draw] += values[method, mi, int(bs[method, draw, unit, j])]/50
                    expected[draw] -= (target[mi] if target.ndim == 2 else target[mi, int(seed[draw, unit, j])])/50
        point = values[method].mean((0, 1))-(target.mean(0) if target.ndim == 2 else target.mean((0, 1)))
        np.testing.assert_allclose(expected, row['bootstrap'], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(point, row['point'], rtol=2e-12, atol=2e-13)
    write(OUT/'statistics_selfcheck.json', {'passed': True, 'contrasts_checked': 15,
        'synthetic_scalar_bootstrap_values_checked': 210, 'source_sha256': sha(__file__),
        'scope': 'Pure contrast/indexing function checked by scalar synthetic enumeration. No fake native/audit records or task data.'})
    print('All15 contrast indexing rules pass independent synthetic scalar enumeration')


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'statistics.json').exists()
    m = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    assert m['complete'] and audit['passed'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    expected = {(method, mi, 4096, seed) for method in range(2) for mi in range(5) for seed in range(10)}
    assert len(m['rows']) == len(audit['rows']) == 100
    assert {(r['method'], r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    recovered = {r['name']: r for r in audit['rows']}
    assert len(recovered) == 100
    for row in m['rows']:
        a = recovered[row['name']]
        assert a['case_sha256'] == row['sha256'] and a['all17500_array_receipts_exact'] and a['all100000_final_logits_exact']
    assert m['native_updates'] == audit['public_updates'] == p['native_updates']
    assert audit['counter_entries_exact'] == p['counter_entries']
    y = np.load(BASE/'test_data.npz')['labels']
    parent = np.load(PARENT/'statistical_arrays.npz')
    base = np.load(BASE/'statistical_arrays.npz')
    word = np.load(WORD/'statistical_arrays.npz')
    precision = np.load(PRECISION/'statistical_arrays.npz')
    refs = references(parent, base, word, precision)
    values = np.empty((2, 5, 10, 2))
    rows = []
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            value = metrics(case['logits'], y)
        values[row['method'], row['model'], row['seed']] = value
        rows.append({**{k: row[k] for k in ['name', 'method', 'model', 'T', 'seed']},
                     'accuracy': float(value[0]), 'mean_CE': float(value[1])})
    bs = np.random.default_rng(p['bootstrap_seed']).integers(0, 10, (2, 5000, 5, 10))
    computed = compute(values, refs, base['bootstrap_models'], bs, parent['bootstrap_method_seeds'],
                       base['bootstrap_seeds'], word['bootstrap_word_seeds'])
    arrays = {'values': values, 'bootstrap_method_seeds': bs}
    contrasts = []
    for row in computed:
        key = f'method{row["method"]}_minus_{row["reference"]}'
        arrays[key] = row['bootstrap']
        contrasts.append({'method_index': row['method'], 'method': METHODS[row['method']],
            'reference': row['reference'], 'array_key': key, 'T': 4096,
            'difference': row['point'].tolist(),
            'descriptive_95_interval': np.quantile(row['bootstrap'], [.025, .975], axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz', **arrays)
    summaries = [{'method_index': method, 'method': METHODS[method], 'T': 4096,
                  'mean': values[method].mean((0, 1)).tolist(), 'model_means': values[method].mean(1).tolist(),
                  'clipped_coordinates': np.sum([r['clipped_coordinates'] for r in m['rows'] if r['method'] == method], axis=0).tolist()}
                 for method in range(2)]
    inputs = [OUT/'manifest.json', OUT/'native_audit.json', OUT/'statistical_arrays.npz',
              PARENT/'statistical_arrays.npz', BASE/'statistical_arrays.npz', WORD/'statistical_arrays.npz',
              PRECISION/'statistical_arrays.npz', BASE/'test_data.npz']
    guard(p)
    write(OUT/'statistics.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'rows': rows,
        'summaries': summaries, 'contrasts': contrasts,
        'artifact_sha256': {str(path): sha(path) for path in inputs},
        'metric_order': ['accuracy_fraction', 'mean_CE_nats'], 'scope': p['statistics']})
    print('All100cases/15contrasts computed after full public replay; independent audit required')


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'run': run}[sys.argv[1]]()
