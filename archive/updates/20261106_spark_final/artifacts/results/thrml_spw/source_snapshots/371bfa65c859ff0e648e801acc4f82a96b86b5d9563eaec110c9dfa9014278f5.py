"""Independent first-order algebra and saved-storage checks for recomputation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b/recomputed_probability_preflight')
    p = json.loads((root / 'protocol.json').read_text())
    r = json.loads((root / 'results.json').read_text())
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert r['status'] == 'all15_cases_forward_first_order_gradient_and_saved_state_checks_passed'
    assert sha(root / 'arrays.npz') == r['arrays_sha256']
    a = np.load(root / 'arrays.npz')
    seen = set()
    checked = 0
    for case in r['cases']:
        kind, variant, shape = case['kind'], case['variant'], case['shape']
        seen.add((kind, variant))
        label = kind + '_' + variant
        theta, lower, span = [a[label + '_' + key] for key in ['theta', 'lower', 'span']]
        assert list(theta.shape) == shape and theta.dtype == np.float32
        codes = np.rint((np.clip(theta.astype(np.float64), -4, 4) + 4) * 4095 / 8)
        h = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        q = np.exp(-np.logaddexp(0, -2 * h.astype(np.float64))).reshape(*span.shape[:-1], -1)
        lo, width = lower.astype(np.float64), span.astype(np.float64)
        mean = (lo + width * q).reshape(shape)
        variance = (width * width * q * (1 - q)).reshape(shape)
        np.testing.assert_allclose(a[label + '_mean'], mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(a[label + '_variance'], variance, rtol=2e-12, atol=2e-12)
        target = torch.linspace(-.5, .5, theta.size).double().numpy().reshape(q.shape)
        derivative = 2 * q * (1 - q)
        expected = (2 * ((lo + width * q) - target) * width * derivative + .7 * width * width * (1 - 2 * q) * derivative) / theta.size
        np.testing.assert_allclose(a[label + '_gradient'], expected.reshape(shape).astype(np.float32), rtol=2e-6, atol=1e-10)
        expected_variance = width * width * (1 - 2 * q) * derivative / theta.size
        np.testing.assert_allclose(a[label + '_variance_gradient'], expected_variance.reshape(shape).astype(np.float32), rtol=2e-6, atol=1e-10)
        saved = case['recomputed_saved_state']
        assert saved['transient_saved_storage_bytes'] == 0
        assert all(v['persistent'] and v['dtype'] == 'torch.float32' for v in saved['saved'])
        assert saved['unique_saved_storage_bytes'] == sum(v['bytes'] for v in saved['saved'])
        old = case['reference_saved_state']
        assert old['transient_saved_storage_bytes'] == sum(v['bytes'] for v in old['saved'] if not v['persistent'])
        assert old['transient_saved_storage_bytes'] >= theta.size * 8
        if kind == 'zero_span':
            assert not np.any(a[label + '_gradient']) and not np.any(a[label + '_variance_gradient'])
        checked += theta.size
    assert seen == {(kind, variant) for kind in ['linear', 'conv', 'embedding', 'elementwise', 'zero_span']
                    for variant in ['initial', 'perturbed', 'endpoint_and_outside']}
    audit = {'status': 'all15_first_order_gradient_laws_and_saved_storage_records_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(root / 'results.json'),
             'gradient_coordinates_checked': checked,
             'scope': 'Independent NumPy first-order declared identity STE algebra and producer saved-storage hook records. No differentiation through rounding or native sampler; no higher-order gradient, total peak-memory, full-model backward/training or language-quality claim.'}
    with (root / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
