"""All full RGB native outcomes and deterministic-control contrasts."""
import json
import numpy as np
from results.thrml_spw.cifar_cnn_native.cohort.common import OUT, CONTROLS, sha, write, save, guard, metrics


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'statistics.json').exists()
    m = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    assert m['complete'] and audit['passed'] and audit['complete'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    expected = {(mi, t, seed) for mi in range(5) for t in p['samples'] for seed in range(10)}
    assert len(m['rows']) == len(audit['rows']) == 150
    assert {(r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    assert {(r['model'], r['T'], r['seed']) for r in audit['rows']} == expected
    assert all(r['all_receipts_and_final_logits_exact'] for r in audit['rows'])
    recovered = {r['name']: r for r in audit['rows']}
    assert len(recovered) == 150
    for row in m['rows']:
        assert recovered[row['name']]['sha256'] == row['sha256']
    assert audit['scheduled_updates'] == 329952000000 and audit['counter_entries'] == 9237375000
    labels = np.load(OUT/'test_data.npz')['labels']
    control = np.load(OUT/'control_logits.npz')
    base = np.empty((10, 5, 2))
    control_rows = []
    for ci, name in enumerate(CONTROLS):
        for mi in range(5):
            v = metrics(control[f'c{ci}_model{mi}'], labels)
            base[ci, mi] = v
            control_rows.append({'control': name, 'model': mi, 'accuracy': float(v[0]), 'mean_CE': float(v[1])})
    values, rows = np.empty((3, 5, 10, 2)), []
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        v = metrics(np.load(row['path'])['logits'], labels)
        bi = p['samples'].index(row['T'])
        values[bi, row['model'], row['seed']] = v
        rows.append({k: row[k] for k in ['name', 'model', 'T', 'seed', 'clipped_coordinates']} |
                    {'accuracy': float(v[0]), 'mean_CE': float(v[1])})
    rng = np.random.default_rng(p['bootstrap_seed'])
    bm, bs = rng.integers(0, 5, (5000, 5)), rng.integers(0, 10, (5000, 5, 10))
    arrays = {'values': values, 'baseline_metrics': base, 'bootstrap_models': bm, 'bootstrap_seeds': bs}
    summaries, contrasts = [], []
    for bi, samples in enumerate(p['samples']):
        summaries.append({'T': samples, 'mean': values[bi].mean((0, 1)).tolist(), 'model_means': values[bi].mean(1).tolist(),
                          'clipped_coordinates': np.sum([r['clipped_coordinates'] for r in m['rows'] if r['T'] == samples], axis=0).tolist()})
        for ci, name in enumerate(CONTROLS):
            delta = values[bi]-base[ci, :, None, :]
            dist = delta[bm[:, :, None], bs].mean((1, 2))
            arrays[f'budget{bi}_control{ci}'] = dist
            contrasts.append({'T': samples, 'budget_index': bi, 'control': name, 'control_index': ci,
                              'difference': delta.mean((0, 1)).tolist(), 'descriptive_95_interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz', **arrays)
    paths = {name: OUT/(name+'.json') for name in ['manifest', 'native_audit', 'controls']}
    paths.update(test_data=OUT/'test_data.npz', control_logits=OUT/'control_logits.npz', statistical_arrays=OUT/'statistical_arrays.npz')
    guard(p)
    write(OUT/'statistics.json', {'artifact_sha256': {name: sha(path) for name, path in paths.items()},
                                 'rows': rows, 'controls': control_rows, 'summaries': summaries, 'contrasts': contrasts,
                                 'control_summary': {name: base[ci].mean(0).tolist() for ci, name in enumerate(CONTROLS)},
                                 'metric_order': ['accuracy_fraction', 'mean_CE_nats'], 'scope': p['statistics']})
    print('All150RGB outcomes/all30contrasts retained; independent verification pending', flush=True)


if __name__ == '__main__':
    main()
