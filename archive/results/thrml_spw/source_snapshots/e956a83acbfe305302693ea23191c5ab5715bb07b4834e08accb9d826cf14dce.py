"""Nominal exact-law first-layer diagnostics; no labels or sampler calls."""
import datetime
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT / 'first_layer_moments'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists()
    training = json.loads((ROOT / 'training/manifest.json').read_text())
    inputs = [ROOT / 'training/manifest.json', ROOT / 'training/validation.npz']
    inputs += [Path(row['path']) for row in training['rows']]
    write(OUT / 'protocol.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': {__file__: sha(__file__)},
        'input_sha256': {str(p): sha(p) for p in inputs},
        'models': 5, 'validation_images': 16, 'bank_size': 8,
        'matched_single_samples': [64, 256, 1024],
        'matched_word_samples': [16, 64, 256],
        'scope': 'Post-CNN descriptive analytic diagnostic, fixed first16validation inputs, no labels/new stochastic draws/quality prediction. Nominal independent Bernoulli stationary law, not empirical finite-PRNG/device behavior. Same kernel shared across all positions/images within each8imagebank. Single finite8bitfield grid and internal4node positional word at finite +/-4 fields. Examine preactivation only before ReLU/maxpool; no inference that independent full-network error follows.'
    })


def run():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert not (OUT / 'completion.json').exists()
    for path, expected in {**p['source_sha256'], **p['input_sha256']}.items():
        assert sha(path) == expected, path
    train = json.loads((ROOT / 'training/manifest.json').read_text())
    x = np.load(ROOT / 'training/validation.npz')['inputs'][:16].astype(float)
    # The separate verifier below uses indexed padding loops, not this view.
    padding = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    patches = np.lib.stride_tricks.sliding_window_view(padding, (3, 3), axis=(-2, -1))
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(2, 8 * 14 * 14, 9)
    bits4 = np.array(list(itertools.product([0., 1.], repeat=4)))
    bits9 = np.array(list(itertools.product([0., 1.], repeat=9)))
    eps = 1 / (1 + np.exp(8.))
    rows = []
    max_check_error = 0.
    for mi, item in enumerate(train['rows']):
        model = np.load(item['path'])
        assert model['activation_span'][0] == 1.
        w = model['conv1_weight'].astype(float).reshape(8, 9)
        for channel, target in enumerate(w):
            c = (target.max() + target.min()) / 2
            r = (target.max() - target.min()) / 2
            assert r > 0
            probability = np.clip((target - c) / (2*r) + .5, eps, 1-eps)
            code = np.rint((np.log(probability / (1-probability)) / 2 + 4) * 255 / 8)
            field = (-4 + code * 8 / 255).astype(np.float32).astype(float)
            single_p = 1 / (1 + np.exp(-2*field))
            means = {'single': c-r+2*r*single_p}
            variances = {'single': 4*r*r*single_p*(1-single_p)}
            quant = np.rint(np.clip((target-c)/(2*r)+.5, 0, 1) * 15).astype(int)
            desired = ((quant[:, None] >> np.arange(4)) & 1).astype(bool)
            word_p = np.where(desired, 1-eps, eps)
            means['word'] = c-r+(2*r/15)*(word_p @ (2.**np.arange(4)))
            variances['word'] = (2*r/15)**2 * np.sum(word_p*(1-word_p)*(4.**np.arange(4)), axis=1)
            for bank in range(2):
                a = np.rint(255*patches[bank])/255
                # Independent coordinate-indexed patches, including real zeros
                # at padding; no activation p-bit or independent position draws.
                direct = []
                for image in x[8*bank:8*(bank+1), 0]:
                    for yy in range(14):
                        for xx in range(14):
                            direct.append([np.rint(255*image[yy+dy, xx+dx])/255 if 0 <= yy+dy < 14 and 0 <= xx+dx < 14 else 0. for dy in [-1, 0, 1] for dx in [-1, 0, 1]])
                np.testing.assert_array_equal(a, direct)
                for method, tlist in [('single', p['matched_single_samples']), ('word', p['matched_word_samples'])]:
                    mean = means[method]
                    var = variances[method]
                    # Sum independent weight contributions. The same a[:,i]
                    # multiplies each weight's fluctuation at every output site.
                    diagonal = (a*a) @ var
                    bias = a @ (mean-target)
                    avg_variance = np.dot(a.mean(0)**2, var)
                    fake_independent = diagonal.sum() / len(a)**2
                    # Enumerate the full512-state single kernel. For the word,
                    # enumerate each weight's16states then add independent
                    # covariance contributions (full36bit product is unnecessary).
                    if method == 'single':
                        mass = np.prod(np.where(bits9, single_p, 1-single_p), axis=1)
                        states = c-r+2*r*bits9
                        mu = mass @ states
                        centered = states-mu
                        cov = (centered.T*mass) @ centered
                    else:
                        mu = np.empty(9)
                        cov = np.zeros((9, 9))
                        states = c-r+(2*r/15)*(bits4 @ (2.**np.arange(4)))
                        for wi in range(9):
                            mass = np.prod(np.where(bits4, word_p[wi], 1-word_p[wi]), axis=1)
                            mu[wi] = mass @ states
                            cov[wi, wi] = mass @ ((states-mu[wi])**2)
                    np.testing.assert_allclose(mu, mean, rtol=2e-13, atol=2e-15)
                    np.testing.assert_allclose(cov, np.diag(var), rtol=2e-11, atol=2e-16)
                    checked_diag = np.einsum('ij,jk,ik->i', a, cov, a)
                    checked_avg = a.mean(0) @ cov @ a.mean(0)
                    np.testing.assert_allclose(checked_diag, diagonal, rtol=2e-11, atol=2e-15)
                    np.testing.assert_allclose(checked_avg, avg_variance, rtol=2e-11, atol=2e-15)
                    max_check_error = max(max_check_error, float(np.max(abs(checked_diag-diagonal))))
                    for t in tlist:
                        rows.append({'model': mi, 'channel': channel, 'bank': bank, 'method': method, 'T': t,
                                     'mean_bias_squared': float(np.mean(bias*bias)),
                                     'mean_preactivation_variance': float(diagonal.mean()/t),
                                     'mean_preactivation_MSE': float(diagonal.mean()/t + np.mean(bias*bias)),
                                     'spatial_image_average_variance': float(avg_variance/t),
                                     'incorrect_independent_site_average_variance': float(fake_independent/t),
                                     'shared_vs_independent_average_variance_ratio': float(avg_variance/fake_independent)})
    assert len(rows) == 480
    summaries = []
    for method, budgets in [('single', p['matched_single_samples']), ('word', p['matched_word_samples'])]:
        for t in budgets:
            subset = [row for row in rows if row['method'] == method and row['T'] == t]
            keys = ['mean_bias_squared', 'mean_preactivation_variance', 'mean_preactivation_MSE', 'spatial_image_average_variance', 'incorrect_independent_site_average_variance']
            summaries.append({'method': method, 'T': t, 'updates_per_weight': t if method == 'single' else 4*t,
                              **{key: float(np.mean([row[key] for row in subset])) for key in keys}})
    write(OUT / 'moments.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'rows': rows, 'summaries': summaries,
                               'enumeration_and_patch_checks_passed': True, 'max_variance_check_error': max_check_error,
                               'new_native_updates': 0, 'scope': p['scope']})
    write(OUT / 'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   'artifact_sha256': {str(OUT/name): sha(OUT/name) for name in ['protocol.json', 'moments.json']},
                                   'status': 'nominal_first_layer_moments_checked_goal_active'})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
