"""All predeclared Gaussian-control contrasts, after full numerical replay."""
import datetime
import json
from pathlib import Path
import sys
import numpy as np
from results.thrml_spw.fashion_cnn_native.gaussian_control.common import (
    BASE, OUT as PARENT, MODES, sha, write, guard, metrics)

OUT = PARENT.parent/'gaussian_statistics_evidence'
CONTRASTS = ['original_mean_Gaussian_minus_native', 'programmed_mean_Gaussian_minus_native',
             'programmed_mean_Gaussian_minus_original_mean_Gaussian',
             'original_mean_Gaussian_minus_FP32', 'programmed_mean_Gaussian_minus_FP32']


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    parent = json.loads((PARENT/'protocol.json').read_text())
    paths = [PARENT/'protocol.json', BASE/'statistical_arrays.npz', BASE/'statistics.json', BASE/'test_data.npz']
    write(OUT/'protocol.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'source_sha256': {str(p): sha(p) for p in Path(__file__).parent.glob('*.py')},
                              'input_sha256': {str(p): sha(p) for p in paths}, 'contrasts': CONTRASTS,
                              'bootstrap_seed': parent['bootstrap_seed'], 'bootstrap_draws': 5000,
                              'scope': parent['statistics'], 'metric_order': ['accuracy_fraction', 'mean_CE_nats']})


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'statistics.json').exists()
    parent = json.loads((PARENT/'protocol.json').read_text())
    guard(parent)
    audit = json.loads((PARENT/'numerical_audit.json').read_text())
    assert audit['passed'] and audit['manifest_sha256'] == sha(PARENT/'manifest.json')
    manifest = json.loads((PARENT/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['rows']) == 300
    labels = np.load(BASE/'test_data.npz')['labels']
    values, rows = np.empty((2, 3, 5, 10, 2)), []
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        v = metrics(np.load(row['path'])['logits'], labels)
        bi = parent['samples'].index(row['T'])
        values[row['mode'], bi, row['model'], row['seed']] = v
        rows.append({k: row[k] for k in ['name', 'mode', 'model', 'T', 'seed']} | {'accuracy': float(v[0]), 'mean_CE': float(v[1])})
    old = np.load(BASE/'statistical_arrays.npz')
    bm, ns = old['bootstrap_models'], old['bootstrap_seeds']
    native, baseline = old['values'], old['baseline_metrics'][0]
    assert bm.shape == (5000, 5) and ns.shape == (5000, 5, 10)
    gs = np.random.default_rng(p['bootstrap_seed']).integers(0, 10, size=(5000, 5, 10))
    arrays = {'values': values, 'bootstrap_models': bm, 'native_bootstrap_seeds': ns, 'Gaussian_bootstrap_seeds': gs}
    summaries, contrasts = [], []
    for bi, samples in enumerate(parent['samples']):
        for mode in range(2):
            summaries.append({'mode': MODES[mode], 'T': samples, 'mean': values[mode, bi].mean((0, 1)).tolist()})
        g = [values[mode, bi][bm[:, :, None], gs].mean((1, 2)) for mode in range(2)]
        n = native[bi][bm[:, :, None], ns].mean((1, 2))
        f = baseline[bm].mean(1)
        distributions = [g[0]-n, g[1]-n, g[1]-g[0], g[0]-f, g[1]-f]
        means = [values[0, bi].mean((0, 1))-native[bi].mean((0, 1)), values[1, bi].mean((0, 1))-native[bi].mean((0, 1)),
                 (values[1, bi]-values[0, bi]).mean((0, 1)), values[0, bi].mean((0, 1))-baseline.mean(0), values[1, bi].mean((0, 1))-baseline.mean(0)]
        for ci, name in enumerate(CONTRASTS):
            arrays[f'b{bi}_c{ci}'] = distributions[ci]
            contrasts.append({'T': samples, 'budget_index': bi, 'contrast_index': ci, 'contrast': name,
                              'difference': means[ci].tolist(), 'descriptive_95_interval': np.quantile(distributions[ci], [.025, .975], axis=0).T.tolist()})
    np.savez_compressed(OUT/'statistical_arrays.npz', **arrays)
    guard(p)
    write(OUT/'statistics.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'manifest_sha256': sha(PARENT/'manifest.json'),
                                  'numerical_audit_sha256': sha(PARENT/'numerical_audit.json'), 'rows': rows,
                                  'statistical_arrays_sha256': sha(OUT/'statistical_arrays.npz'), 'summaries': summaries,
                                  'contrasts': contrasts, 'scope': p['scope'], 'new_native_updates': 0})
    print('All300Gaussian outcomes and15contrasts retained; independent statistics check pending', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
