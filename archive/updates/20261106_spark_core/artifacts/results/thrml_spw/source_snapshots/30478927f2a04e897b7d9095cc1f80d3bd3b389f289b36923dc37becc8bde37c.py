"""Separate NumPy contractions versus packed Torch convolutions, validation only."""
import datetime
import json
import math
import numpy as np
import torch

from results.thrml_spw.fashion_cnn_native.gaussian_control.common import (
    ROOT, TRAIN, OUT, PREFIXES, MODES, sha, write, guard, moments, weights_for, torch_forward)


def numpy_forward(x, model, weights):
    h = np.asarray(x, float)
    stages, codes, clipping = [], [], []
    banks, images = h.shape[:2]
    for li, prefix in enumerate(PREFIXES):
        span = float(model['activation_span'][li])
        clipping.append(int(np.count_nonzero((h < 0) | (h > span))))
        q = np.rint(255*np.clip(h/span, 0, 1)).astype(np.uint8)
        codes.append(q)
        h = span*q.astype(float)/255
        w = weights[li]
        bias = model[prefix+'_bias'].astype(float)
        if li < 2:
            _, _, channels, height, width = h.shape
            padded = np.pad(h, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
            view = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(-2, -1))
            patches = view.transpose(0, 1, 3, 4, 2, 5, 6).reshape(banks, images*height*width, channels*9)
            h = patches@w.reshape(banks, len(bias), -1).transpose(0, 2, 1)+bias
            h = np.maximum(h.reshape(banks, images, height, width, len(bias)).transpose(0, 1, 4, 2, 3), 0)
            h = h[:, :, :, :2*(height//2), :2*(width//2)].reshape(banks, images, len(bias), height//2, 2, width//2, 2).max((4, 6))
        else:
            h = h.reshape(banks, images, -1)@w.transpose(0, 2, 1)+bias
        stages.append(h)
    return h, stages, codes, np.array(clipping, np.int64)


def main():
    assert not (OUT/'preflight.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    train = json.loads((TRAIN/'manifest.json').read_text())
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 1, 14, 14)
    rows, variance_entries = [], 0
    with torch.no_grad():
        for mi in range(5):
            model = np.load(train['rows'][mi]['path'])
            for prefix in PREFIXES:
                nominal, variance, probs, commands = moments(model, prefix)
                flat = model[prefix+'_weight'].astype(float).reshape(len(nominal), -1)
                for output, weights in enumerate(flat):
                    lo, hi = min(weights), max(weights)
                    for wi, value in enumerate(weights):
                        q = min(1/(1+math.exp(-8)), max(1/(1+math.exp(8)), (value-lo)/(hi-lo)))
                        code = round((math.log(q/(1-q))/2+4)*255/8)
                        assert code == commands[output, wi]
                        field = float(np.float32(-4+code*8/255))
                        exponential = math.exp(2*field)
                        prob = exponential/(1+exponential)
                        mu = (1-prob)*lo+prob*hi
                        var = (1-prob)*(lo-mu)**2+prob*(hi-mu)**2
                        np.testing.assert_allclose([mu, var], [nominal.reshape(len(nominal), -1)[output, wi], variance.reshape(len(nominal), -1)[output, wi]], rtol=5e-11, atol=2e-15)
                        variance_entries += 1
            for samples in p['samples']:
                for seed in [0, 1]:
                    for mode in range(2):
                        weights = [weights_for(model, mi, samples, seed, li, 0, 2, mode) for li in range(3)]
                        actual = torch_forward(x, model, weights)
                        expected = numpy_forward(x, model, weights)
                        for li in range(3):
                            np.testing.assert_array_equal(actual[2][li], expected[2][li])
                            np.testing.assert_allclose(actual[1][li], expected[1][li], rtol=2e-12, atol=2e-11)
                        np.testing.assert_array_equal(actual[3], expected[3])
                        rows.append({'model': mi, 'T': samples, 'seed': seed, 'mode': MODES[mode], 'maximum_logit_error': float(np.max(abs(actual[0]-expected[0])))})
    assert len(rows) == 60 and variance_entries == 13320
    guard(p)
    write(OUT/'preflight.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               'protocol_sha256': sha(OUT/'protocol.json'), 'rows': rows,
                               'weight_variances_independently_checked': variance_entries,
                               'scope': 'All5starting models/2centering modes/3budgets/2seeds, first16validation inputs, all activation codes/stage outputs/clipping. Separate scalar two-outcome mean/variance calculation on all13320weights. No labels/task metrics/native calls.'})
    print('Gaussian control preflight:60paths and13320variances checked', flush=True)


if __name__ == '__main__':
    main()
