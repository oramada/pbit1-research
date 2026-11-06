"""Audit conditional error covariance from retained sufficient statistics."""
import json
import numpy as np
from .prepare_v1 import OUT, sha


def main():
    # Validate moment subtraction against direct per-input sample covariances.
    random = np.random.default_rng(2026112701).normal(size=(5, 7, 3))
    cross = np.einsum('sni,snj->ij', random, random)
    summed = random.sum(0)
    computed = (cross - summed.T @ summed / 5) / (7 * 4)
    reference = np.mean([np.cov(random[:, n], rowvar=False, ddof=1) for n in range(7)], axis=0)
    np.testing.assert_allclose(computed, reference, rtol=1e-13)
    root = OUT / 'lfm3_error_covariance_v2'
    result = json.loads((root / 'results.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    assert result['protocol_sha256'] == sha(root / 'protocol.json')
    assert protocol['training_result_sha256'] == sha(OUT / 'lfm3_training_depth_v1/results.json')
    assert protocol['seeds'] == list(range(2026111801, 2026111806))
    assert [x['seed'] for x in result['checked']] == protocol['seeds']
    assert all(x['weight_count_hashes_checked'] == 286 and x['replayed_targets'] == 8192 for x in result['checked'])
    with np.load(root / 'covariances.npz') as f:
        covariance = f['covariance'].copy()
        cross = f['cross_sums'].copy()
        outer = f['mean_outer_products'].copy()
        channels = f['channels'].copy()
        assert int(f['positions']) == 8224 and int(f['draws']) == 5
    assert covariance.shape == cross.shape == outer.shape == (2, 24, 64, 64)
    np.testing.assert_allclose(covariance, (cross-outer)/(8224*4), rtol=1e-12, atol=1e-18)
    expected_diagonal = np.zeros((2, 24, 64))
    for seed in protocol['seeds']:
        with np.load(OUT / f'lfm3_training_depth_v1/selected_seed{seed}/channels.npz') as f:
            for group in range(2):
                for layer in range(24):
                    expected_diagonal[group, layer] += f['error_SSE'][layer, channels[group, layer]]
    np.testing.assert_allclose(np.diagonal(cross, axis1=2, axis2=3), expected_diagonal, rtol=1e-12, atol=1e-18)
    for row in result['summaries']:
        matrix = covariance[row['group'], row['layer']]
        eigen = np.linalg.eigvalsh(matrix)
        assert eigen.min() >= -1e-10 * max(eigen[-1], 1e-30)
        std = np.sqrt(np.maximum(np.diag(matrix), 0))
        corr = matrix / np.maximum(np.outer(std, std), 1e-30)
        np.testing.assert_allclose(np.mean(np.abs(corr[np.triu_indices(64, 1)])), row['mean_absolute_error_correlation'], rtol=1e-12)
        np.testing.assert_allclose(np.trace(matrix)**2/np.sum(matrix*matrix), row['covariance_participation_ratio'], rtol=1e-12)
        np.testing.assert_allclose(eigen[-1]/eigen.sum(), row['largest_eigenvalue_fraction'], rtol=1e-12)
    output = {'status': 'all48_conditional_covariance_matrices_and_original_trace_diagonals_audited', 'source_sha256': sha(__file__),
              'result_sha256': sha(root / 'results.json'), 'direct_covariance_fixture': True, 'scope': protocol['scope']}
    (OUT / 'lfm3_error_covariance_v2_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))

if __name__ == '__main__':
    main()
