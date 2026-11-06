"""Separate scalar-outcome enumeration of every retained first-layer moment."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT / 'first_layer_moments'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT / 'independent_audit.json').exists()
    closed = json.loads((OUT / 'completion.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for mapping in [closed['artifact_sha256'], protocol['source_sha256'], protocol['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path
    result = json.loads((OUT / 'moments.json').read_text())
    training = json.loads((ROOT / 'training/manifest.json').read_text())
    inputs = np.load(ROOT / 'training/validation.npz')['inputs'][:16, 0].astype(float)
    indexed = {}
    for bank in range(2):
        indexed[bank] = np.array([[round(255*image[y+dy, x+dx])/255 if 0 <= y+dy < 14 and 0 <= x+dx < 14 else 0.
                                  for dy in range(-1, 2) for dx in range(-1, 2)]
                                 for image in inputs[bank*8:(bank+1)*8] for y in range(14) for x in range(14)])
    expected = {(m, c, b, method, t) for m in range(5) for c in range(8) for b in range(2)
                for method, budgets in [('single', [64, 256, 1024]), ('word', [16, 64, 256])]
                for t in budgets}
    assert len(result['rows']) == len(expected) == 480
    assert {(r['model'], r['channel'], r['bank'], r['method'], r['T']) for r in result['rows']} == expected
    cache = {}
    for row in result['rows']:
        key = row['model'], row['channel'], row['method']
        if key not in cache:
            w = np.load(training['rows'][key[0]]['path'])['conv1_weight'][key[1]].astype(float).ravel()
            lo, hi = min(w), max(w)
            mean, var = [], []
            for weight in w:
                q = (weight-lo)/(hi-lo)
                if key[2] == 'single':
                    q = min(1/(1+np.exp(-8.)), max(1/(1+np.exp(8.)), q))
                    field = float(np.float32(-4 + round((np.log(q/(1-q))/2+4)*255/8)*8/255))
                    prob = 1/(1+np.exp(-2*field))
                    states = [(lo, 1-prob), (hi, prob)]
                else:
                    code = round(q*15)
                    probabilities = [1/(1+np.exp(-8. if code & (1 << k) else 8.)) for k in range(4)]
                    states = []
                    for bits in itertools.product([0, 1], repeat=4):
                        probability = np.prod([probabilities[k] if bits[k] else 1-probabilities[k] for k in range(4)])
                        states.append((lo+(hi-lo)*sum(bits[k]*(1 << k) for k in range(4))/15, probability))
                mu = sum(value*mass for value, mass in states)
                variance = sum((value-mu)**2*mass for value, mass in states)
                mean.append(mu)
                var.append(variance)
            cache[key] = w, np.array(mean), np.array(var)
        w, mean, var = cache[key]
        a = indexed[row['bank']]
        bias = sum(a[:, j]*(mean[j]-w[j]) for j in range(9))
        diag = sum(a[:, j]**2*var[j] for j in range(9))
        aggregate = sum(float(a[:, j].mean())**2*var[j] for j in range(9))
        wrong = sum(diag)/len(diag)**2
        t = row['T']
        calculated = {'mean_bias_squared': np.mean(bias**2), 'mean_preactivation_variance': np.mean(diag)/t,
                      'mean_preactivation_MSE': np.mean(diag)/t+np.mean(bias**2),
                      'spatial_image_average_variance': aggregate/t,
                      'incorrect_independent_site_average_variance': wrong/t,
                      'shared_vs_independent_average_variance_ratio': aggregate/wrong}
        for name, value in calculated.items():
            np.testing.assert_allclose(value, row[name], rtol=5e-11, atol=3e-15)
    for summary in result['summaries']:
        rows = [r for r in result['rows'] if r['method'] == summary['method'] and r['T'] == summary['T']]
        for name in ['mean_bias_squared', 'mean_preactivation_variance', 'mean_preactivation_MSE', 'spatial_image_average_variance', 'incorrect_independent_site_average_variance']:
            np.testing.assert_allclose(np.mean([r[name] for r in rows]), summary[name], rtol=2e-13, atol=2e-15)
    output = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'parent_completion_sha256': sha(OUT/'completion.json'),
              'moments_sha256': sha(OUT/'moments.json'), 'rows_checked': 480, 'scalar_moments_checked': 2880,
              'new_native_updates': 0, 'scope': 'Independent indexed-patch/scalar outcome enumeration, all retained moments and summaries checked. Nominal Bernoulli mathematics, not empirical raw-state recovery or a full-network quality prediction.'}
    (OUT / 'independent_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print('All480 conditional first-layer moment rows independently checked')


if __name__ == '__main__':
    main()
