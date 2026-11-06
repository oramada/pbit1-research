"""Independent NumPy readout, declared STE gradient and binomial moment audit."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'probability_operator_preflight'
    protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    prior = root / 'program_export_preflight'
    assert sha(prior / 'independent_audit.json') == protocol['reference_audit_sha256']
    assert sha(prior / 'results.json') == protocol['reference_results_sha256']
    assert result['status'] == 'all34_ACT12_operator_public_draw_outputs_and_gradients_saved'
    assert len(result['cases']) == 34
    grad_coordinates = native_cases = 0
    for rec in result['cases']:
        path = Path(rec['path'])
        assert sha(path) == rec['file_sha256']
        assert sha(Path(rec['reference_output'])) == rec['reference_output_sha256']
        a = np.load(path)
        codes = a['codes']
        rows = codes.shape[0] if codes.ndim > 1 else 1
        width = codes.size // rows
        size = min(128, width)
        shape = (rows, width // size, size)
        h = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        p = np.exp(-np.logaddexp(0, -2 * h.astype(np.float64))).reshape(shape)
        if rec['mode'] == 'native':
            counts = a['counts']
            assert counts.shape == codes.shape and counts.dtype == np.int64
            assert counts.min() >= 0 and counts.max() <= rec['samples']
            fraction = counts.reshape(shape).astype(np.float64) / rec['samples']
            native_cases += 1
        else:
            assert a['counts'].size == 0
            fraction = p
        w = (a['lower'].astype(np.float64) + a['span'].astype(np.float64) * fraction).reshape(codes.shape)
        np.testing.assert_allclose(a['weights'], w, rtol=2e-12, atol=2e-12)
        x, upstream = a['x'], a['upstream']
        scale = float(np.abs(x).max()) / 2047 if np.any(x) else 1.
        assert float(a['scale']) == scale
        integer = np.rint(x / scale).clip(-2047, 2047).astype(np.int32)
        np.testing.assert_array_equal(a['integers'], integer)
        value = integer.astype(np.float64) * scale
        kind = rec['kind']
        if kind == 'linear':
            y = value @ w.T
            grad_w = upstream.T @ value
            grad_x = upstream @ w
        else:
            layout = w[:, 0, :].T if kind == 'conv' else w.reshape(1, -1)
            y = (value * layout[None]).sum(1)
            grad_x = upstream[:, None, :] * layout[None]
            product = (upstream[:, None, :] * value).sum(0)
            grad_w = product.T[:, None, :] if kind == 'conv' else product.reshape(w.shape)
        gradient = (grad_w.reshape(shape) * a['span'].astype(np.float64) * 2 * p * (1 - p)).reshape(w.shape).astype(np.float32)
        np.testing.assert_allclose(a['output'], y, **protocol['output_tolerance'])
        np.testing.assert_allclose(a['output'], np.load(rec['reference_output']), **protocol['output_tolerance'])
        np.testing.assert_allclose(a['theta_gradient'], gradient, **protocol['gradient_tolerance'])
        np.testing.assert_allclose(a['input_gradient'], grad_x.astype(x.dtype), **protocol['gradient_tolerance'])
        grad_coordinates += gradient.size
    assert native_cases == 17
    assert sha(Path(result['law_path'])) == result['law_sha256']
    law = np.load(result['law_path'])
    fields = law['codes'].astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
    p = np.exp(-np.logaddexp(0, -2 * fields.astype(np.float64)))
    np.testing.assert_allclose(law['probabilities'], p, rtol=2e-14, atol=2e-14)
    trials = protocol['binomial_trials']
    diagnostics = []
    for samples in protocol['binomial_samples']:
        counts = law[f'counts_{samples}']
        assert counts.shape == (trials, len(p)) and counts.dtype == np.int64
        assert counts.min() >= 0 and counts.max() <= samples
        values = counts.astype(np.float64) / samples
        variance = p * (1 - p) / samples
        fourth = (samples * p * (1 - p) * (1 - 6 * p * (1 - p)) + 3 * (samples * p * (1 - p)) ** 2) / samples ** 4
        se_mean = np.sqrt(variance / trials)
        se_variance = np.sqrt((fourth - (trials - 3) / (trials - 1) * variance ** 2) / trials)
        mean_z = np.abs(values.mean(0) - p) / se_mean
        variance_z = np.abs(values.var(0, ddof=1) - variance) / se_variance
        assert mean_z.max() < protocol['moment_test_standard_errors']
        assert variance_z.max() < protocol['moment_test_standard_errors']
        diagnostics.append({'samples': samples, 'max_mean_standard_errors': float(mean_z.max()),
                            'max_variance_standard_errors': float(variance_z.max())})
    audit = {'status': 'all34_ACT12_public_draw_operator_and_STE_laws_and_binomial_moments_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'cases': 34, 'native_public_draw_cases': native_cases, 'control_gradient_coordinates': grad_coordinates,
             'binomial_moment_checks': diagnostics,
             'scope': 'Independent saved FP64 affine readout parity to preserved official THRML counts and exact ACT12 integer inputs; NumPy declared projected expectation-STE/control and identity-STE/input gradient algebra. Binomial moments tested at T1/16/1024; not identity to finite-precision JAX marginals, unbiased sampler gradients, full-model parity, adaptation or hardware advantage.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
