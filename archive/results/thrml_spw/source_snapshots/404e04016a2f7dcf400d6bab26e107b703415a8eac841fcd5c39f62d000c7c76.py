"""Bind synthetic-tested uncertainty arithmetic to reserved article metadata.

Reads article IDs and offsets only; no token arrays or model predictions loaded.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    directory = root / 'holdout_data'
    manifest = json.loads((directory / 'manifest.json').read_text())
    original_audit = json.loads((directory / 'independent_audit.json').read_text())
    assert original_audit['status'] == 'all_holdout_contexts_article_exclusions_and_reservations_verified'
    fixture_path = root / 'holdout_uncertainty_preflight.json'
    fixture = json.loads(fixture_path.read_text())
    assert fixture['status'] == 'synthetic_article_grouping_exact_resample_replay_and_common_seed_noise_verified'
    for name, digest in fixture['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert manifest['status'] == 'article_separated_holdout_frozen_no_model_scores'
    tokens_path = directory / 'tokens.npz'
    assert sha(tokens_path) == manifest['tokens_sha256']
    columns = {}
    with np.load(tokens_path) as arrays:
        for label, count, repeats, length in [('short', 24, 4, 65), ('long', 16, 1, 257)]:
            ids, offsets = arrays[label + '_article_ids'], arrays[label + '_offsets']
            expected = np.asarray(manifest['cohorts'][label]['article_ids'], dtype=np.int64)
            assert ids.shape == offsets.shape == (count * repeats,)
            assert len(expected) == count
            np.testing.assert_array_equal(ids, np.repeat(expected, repeats))
            np.testing.assert_array_equal(offsets, manifest['cohorts'][label]['selected_offsets'])
            assert repeats * (length - 1) == 256
            assert manifest['cohorts'][label]['predicted_tokens'] == count * 256
            for article, start in zip(ids, offsets, strict=True):
                record = next(a for a in manifest['articles'] if a['article_id'] == int(article))
                assert record['token_start'] <= int(start) and int(start) + length <= record['token_end']
            columns[label] = expected.tolist()
    assert not set(columns['short']) & set(columns['long'])
    assert manifest['total_predicted_targets'] == 10240
    result = {'status': 'reserved_article_order_and_256_target_cluster_geometry_verified_without_scoring',
        'source_sha256': sha(Path(__file__)), 'manifest_sha256': sha(directory / 'manifest.json'),
        'holdout_audit_sha256': sha(directory / 'independent_audit.json'), 'fixture_sha256': sha(fixture_path),
        'article_column_ids': columns, 'pooled_token_weights': [.6, .4], 'targets_per_article': 256,
        'scope': 'Metadata-only alignment of uncertainty component to reserved design; archive hash read but token arrays/model scores not loaded. Synthetic correctness is not coverage calibration. Treatment, forward implementation, seed plan, analysis protocol and independent scoring audit still required before consuming reserved outcomes.'}
    with (root / 'holdout_uncertainty_geometry_audit.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
