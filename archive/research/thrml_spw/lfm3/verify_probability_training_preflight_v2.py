"""Independent STE-gradient, finite-command, marginal-law and public-state audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'probability_training_preflight'
    protocol = json.loads((out / 'protocol.json').read_text())
    probe=torch.tensor([-4.00001,-4.,0.,4.,4.00001],dtype=torch.float64,requires_grad=True)
    probe.clamp(-4,4).sum().backward()
    assert probe.grad.tolist()==[0.,0.,1.,0.,0.], 'Framework boundary convention changed; inspect explicitly.'
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    source = root / 'group128_preflight/input_0.npz'
    assert sha(source) == protocol['input_sha256']
    training = json.loads((out / 'training_results.json').read_text())
    assert sha(out / 'training_arrays.npz') == training['arrays_sha256']
    a = np.load(out / 'training_arrays.npz')
    original = np.load(source)
    np.testing.assert_array_equal(a['weights'], original['w'][:128, :8])
    np.testing.assert_array_equal(a['inputs'], original['x'][:, :128])
    scale = float(np.max(np.abs(a['inputs']))) / 2047
    integers = np.rint(a['inputs'] / scale).clip(-2047, 2047).astype(np.int64)
    np.testing.assert_array_equal(integers, a['input_integers'])
    qx = integers * scale
    target = qx @ a['weights']
    np.testing.assert_allclose(a['target'], target, rtol=2e-12, atol=2e-12)
    lo = a['weights'].min(0)[None, :]
    span = (a['weights'].max(0) - a['weights'].min(0))[None, :]
    np.testing.assert_array_equal(a['lower'].reshape(1, 8), lo)
    np.testing.assert_array_equal(a['span'].reshape(1, 8), span)
    native = json.loads((out / 'native_results.json').read_text())
    assert native['status'] == 'both_conditions_full_public_replay_exact'
    history = training['history']
    assert [v['step'] for v in history] == list(range(201))
    statistics = []
    trial_mses = {}
    artifacts = {str(p): sha(p) for p in [out / 'protocol.json', out / 'training_results.json', out / 'training_arrays.npz', out / 'native_results.json', out / 'mean_reference.npy']}
    for label, step in [('before', 0), ('after', 200)]:
        theta = a[label + '_theta'].reshape(128, 8)
        assert np.max(np.abs(theta)) <= 4
        codes = np.rint((theta + 4) * 4095 / 8).astype(np.int32)
        fields = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        np.testing.assert_array_equal(codes, a[label + '_codes'].reshape(128, 8))
        np.testing.assert_array_equal(fields, a[label + '_fields'].reshape(128, 8))
        p = np.exp(-np.logaddexp(0, -2 * fields.astype(np.float64)))
        wmean = lo + span * p
        wvariance = span * span * p * (1 - p)
        output_mean = qx @ wmean
        output_variance = (qx * qx) @ wvariance / 16
        np.testing.assert_allclose(a[label + '_mean'], output_mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(a[label + '_variance'], output_variance, rtol=2e-12, atol=2e-12)
        expected = float(np.mean((output_mean - target) ** 2 + output_variance))
        np.testing.assert_allclose(history[step]['expected_MSE'], expected, rtol=2e-12, atol=2e-12)
        if label == 'before':
            np.testing.assert_allclose(np.load(out / 'mean_reference.npy'), output_mean, rtol=2e-12, atol=2e-12)
            # Closed-form chain rule for the declared identity-rounding STE,
            # not a finite-difference assertion about nondifferentiable rounding.
            derivative = 2 * p * (1 - p)
            mean_part = qx.T @ (2 * (output_mean - target) / target.size)
            variance_part = (qx * qx).sum(0)[:, None] / (target.size * 16)
            gradient = (mean_part * span * derivative + variance_part * span * span * (1 - 2 * p) * derivative) * ((theta > -4) & (theta < 4))
            np.testing.assert_allclose(a['initial_gradient'].reshape(128, 8), gradient, rtol=2e-10, atol=2e-13)
        raw_path = out / (label + '_native.npz')
        raw = np.load(raw_path)
        states = raw['states']
        assert states.shape == (500, 16, 128, 8) and states.dtype == np.bool_
        # Sum public states first, then perform a separate integer contraction.
        public_total = np.einsum('bi,nio->nbo', integers, states.sum(1, dtype=np.int64))
        np.testing.assert_array_equal(public_total, raw['total'].astype(np.int64))
        outputs = (public_total.astype(np.float64) / 16 * span + integers.sum(1)[None, :, None] * lo) * scale
        np.testing.assert_allclose(raw['outputs'], outputs, rtol=2e-12, atol=2e-12)
        mse = np.mean((outputs - target[None, :, :]) ** 2, axis=(1, 2))
        trial_mses[label] = mse
        row = next(v for v in native['conditions'] if v['condition'] == label)
        se = float(mse.std(ddof=1) / np.sqrt(500))
        np.testing.assert_allclose(row['empirical_mean_MSE'], mse.mean(), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(row['MSE_standard_error_across500_trials'], se, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(row['per_seed_mean_MSE'], mse.reshape(5, 100).mean(1), rtol=2e-12, atol=2e-12)
        assert row['retained_node_evaluations'] == 8192000 and row['warmup_node_evaluations'] == 512000
        z = (float(mse.mean()) - expected) / se
        assert abs(z) < 6, (label, z)
        statistics.append({'condition': label, 'expected_MSE': expected, 'empirical_MSE': float(mse.mean()), 'MSE_standard_error': se, 'MSE_theory_z': z})
        artifacts[str(raw_path)] = sha(raw_path)
    difference = trial_mses['after'] - trial_mses['before']
    assert history[-1]['expected_MSE'] < history[0]['expected_MSE']
    audit = {'status': 'controlled_training_gradient_commands_mean_law_and_all_public_states_verified',
             'source_sha256': sha(Path(__file__)), 'torch_version':torch.__version__, 'clamp_gradient_probe':probe.grad.tolist(), 'initial_exact_boundary_fields':int(np.sum(np.abs(a['before_theta'])==4)), 'prior_audit_failure_preserved':True, 'artifacts_sha256': artifacts,
             'statistics': statistics, 'paired_after_minus_before_MSE': float(difference.mean()),
             'paired_MSE_standard_error_across500_trials': float(difference.std(ddof=1) / np.sqrt(500)),
             'exact_integer_accumulator_entries': 128000, 'public_binary_observations': 16384000,
             'scope': 'Controlled1024-weight/16-input-position projection slice. Exact declared STE gradient algebra (local clamp subgradient zero at exact limits) and before/after native/public parity; nominal analytic MSE checked with MC empirical standard error. Five100-trial streams with shared before/after keys. No full LFM3 training, language improvement, unbiased sampler gradient, BF16 equivalence, unseen-input generalization or physical benefit claimed.'}
    with (out / 'independent_audit_v2.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
