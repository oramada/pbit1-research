"""Independent full-coordinate moments via endpoint laws and16-state words."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'word_moments'
    protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    assert sha(Path(__file__).with_name('word_moments.py')) == protocol['source_sha256'] == result['source_sha256']
    assert sha(root / 'meta_inventory.json') == protocol['inventory_sha256']
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    inventory = json.loads((root / 'meta_inventory.json').read_text())
    assert [(p['name'], p['shape']) for p in inventory['parameters']] == [(p['name'], p['shape']) for p in result['layers']]
    # Enumerate the complete four-bit error channel rather than using the
    # producer's weighted Bernoulli variance formula.
    error = np.exp(-np.logaddexp(0., 8.))
    values = np.arange(16, dtype=np.float64) / 15
    word_mean = np.zeros(16)
    word_variance = np.zeros(16)
    for intended in range(16):
        masses = np.asarray([error ** ((intended ^ observed).bit_count()) *
                             (1 - error) ** (4 - (intended ^ observed).bit_count())
                             for observed in range(16)])
        np.testing.assert_allclose(masses.sum(), 1, atol=1e-15, rtol=1e-15)
        word_mean[intended] = masses @ values
        word_variance[intended] = masses @ ((values - word_mean[intended]) ** 2)
    decoded = np.arange(4096, dtype=np.float32) * np.float32(8 / 4095) - np.float32(4)
    probability = 1 / (1 + np.exp(-2 * decoded.astype(np.float64)))
    totals = {m: {'weights': 0, 'groups': 0, 'squared_bias_sum': 0., 'one_realization_variance_sum': 0.}
              for m in protocol['methods']}
    checked = 0
    with safe_open(str(checkpoint), framework='pt') as f:
        for parameter, saved in zip(inventory['parameters'], result['layers'], strict=True):
            name, shape = parameter['name'], parameter['shape']
            vector = len(shape) == 1
            width = shape[0] if vector else int(np.prod(shape[1:]))
            rows = 1 if vector else shape[0]
            group_size = min(128, width)
            sums = {m: {k: 0 for k in totals[m]} for m in totals}
            for start in range(0, rows, 128):
                tensor = f.get_tensor(name) if vector else f.get_slice(name)[start:min(start + 128, rows)]
                weights = tensor.float().numpy().astype(np.float64).reshape(-1, group_size)
                low, high = weights.min(1)[:, None], weights.max(1)[:, None]
                width_of_range = high - low
                p = np.divide(weights - low, width_of_range, out=np.full_like(weights, .5), where=width_of_range != 0)
                clipped = np.maximum(1e-12, np.minimum(1 - 1e-12, p))
                h = np.maximum(-4, np.minimum(4, (np.log(clipped) - np.log1p(-clipped)) / 2))
                code = np.rint((h + 4) / 8 * 4095).astype(np.int64)
                q = probability[code]
                expected_single = (1 - q) * low + q * high
                single_bias = (1 - q) * (low - weights) + q * (high - weights)
                single_variance = (1 - q) * (low - expected_single) ** 2 + q * (high - expected_single) ** 2
                target_word = np.rint(np.maximum(0, np.minimum(1, p)) * 15).astype(np.int64)
                word_bias = low + width_of_range * word_mean[target_word] - weights
                word_var = width_of_range ** 2 * word_variance[target_word]
                for method, bias, variance in [('single12', single_bias, single_variance),
                                               ('word4_endpoint', word_bias, word_var)]:
                    sums[method]['weights'] += weights.size
                    sums[method]['groups'] += low.size
                    sums[method]['squared_bias_sum'] += float(np.sum(bias * bias))
                    sums[method]['one_realization_variance_sum'] += float(np.sum(variance))
            for method in sums:
                for key, expected in sums[method].items():
                    actual = saved['moments'][method][key]
                    if key in ['weights', 'groups']:
                        assert actual == expected
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
                    totals[method][key] += expected
                    checked += 1
            (out / 'audit_progress.json').write_text(json.dumps({'last_tensor': name, 'checked_tensor_moments': checked}) + '\n')
    for method, sums in totals.items():
        actual = result['summary'][method]
        for key, value in sums.items():
            np.testing.assert_allclose(actual[key], value, rtol=2e-10, atol=2e-12)
        assert sums['weights'] == 1728801536 and sums['groups'] == 13531430
        b = sums['squared_bias_sum'] / sums['weights']
        v = sums['one_realization_variance_sum'] / sums['weights']
        np.testing.assert_allclose(actual['mean_squared_bias'], b, rtol=2e-10, atol=2e-15)
        np.testing.assert_allclose(actual['mean_one_realization_variance'], v, rtol=2e-10, atol=2e-15)
        for budget in protocol['matched_evaluations_per_weight']:
            samples = budget if method == 'single12' else budget // 4
            expected = b + v / samples
            np.testing.assert_allclose(actual['scalar_MSE_at_matched_evaluations'][str(budget)], expected, rtol=2e-10, atol=2e-15)
    single, word = [totals[m] for m in ['single12', 'word4_endpoint']]
    crossing = (single['one_realization_variance_sum'] - 4 * word['one_realization_variance_sum']) / (word['squared_bias_sum'] - single['squared_bias_sum'])
    np.testing.assert_allclose(result['formal_scalar_MSE_crossing_evaluations_per_weight'], crossing, rtol=2e-10, atol=2e-12)
    audit = {'status': 'all_full_checkpoint_scalar_moments_verified_by_endpoint_and_16word_laws',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'tensor_moment_checks': checked, 'unique_weights_per_method': 1728801536,
             'scope': 'Independent full-coordinate marginal expectation/variance laws, different chunking and exhaustive16state word channel. Nominal stationary sigmoid laws in float64; finite simulator probability arithmetic is float32. Not empirical MC, neural quality, correlation, BF16 equivalence or hardware performance.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
