"""Independent complete-target coverage fixture; never reads reserved tokens."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .holdout_batch_layout import batch_plan, assemble_losses
from .holdout_uncertainty import article_means


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    manifest_path = root / 'holdout_data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    plan = batch_plan(manifest)
    # Independent literal geometry oracle, with one unique synthetic value per
    # target. No token IDs, predictions, or model weights are used.
    assert len(plan) == 16
    assert [p['input_shape'] for p in plan] == [[8, 65]] * 12 + [[4, 257]] * 4
    targets, rows, records = [], [], []
    for index, spec in enumerate(plan):
        length, count, start = (64, 8, index * 8) if index < 12 else (256, 4, (index - 12) * 4)
        base = start * length + (0 if index < 12 else 6144)
        expected = np.arange(base, base + count * length, dtype=np.float64).reshape(count, length)
        assert spec['global_target_start'] == base
        assert spec['global_target_stop'] == base + expected.size
        label = 'short' if index < 12 else 'long'
        repeats = 4 if index < 12 else 1
        assert spec['article_ids_by_window'] == [manifest['cohorts'][label]['article_ids'][w // repeats] for w in range(start, start + count)]
        records.append({'batch_index': index, 'token_NLL': expected})
        targets.extend(expected.ravel().astype(int).tolist())
        rows.extend(zip(spec['article_ids_by_window'], spec['source_token_offsets']))
    assert targets == list(range(10240)) and len(set(rows)) == 112
    short, long = assemble_losses(plan, records[::-1])
    np.testing.assert_array_equal(short, np.arange(6144).reshape(96, 64))
    np.testing.assert_array_equal(long, np.arange(6144, 10240).reshape(16, 256))
    # Exercise all five MC rows and every target's article ownership.
    s = np.stack([short + seed * 10240 for seed in range(5)])
    l = np.stack([long + seed * 10240 for seed in range(5)])
    sa, la = article_means(s, l)
    for seed in range(5):
        for article in range(40):
            expected = seed * 10240 + article * 256 + 127.5
            assert (sa[seed, article] if article < 24 else la[seed, article - 24]) == expected
    rejected = []
    bad_records = [records[:-1], records + [records[0]],
        [{'batch_index': True, 'token_NLL': records[0]['token_NLL']}] + records[1:],
        [{'batch_index': 16, 'token_NLL': records[0]['token_NLL']}] + records[1:],
        [{'batch_index': 0, 'token_NLL': np.zeros((8, 63))}] + records[1:],
        [{'batch_index': 0, 'token_NLL': np.full((8, 64), np.nan)}] + records[1:],
        [{'batch_index': 0, 'token_NLL': records[0]['token_NLL'], 'extra': 1}] + records[1:]]
    for index, bad in enumerate(bad_records):
        try:
            assemble_losses(plan, bad)
        except ValueError:
            rejected.append('records_' + str(index))
        else:
            raise AssertionError('Invalid records accepted')
    bad_manifests = []
    for mutation in ['short_only', 'duplicate_article', 'cross_stratum_article', 'missing_offset', 'overlapping_windows', 'outside_article']:
        bad = copy.deepcopy(manifest)
        if mutation == 'short_only': bad['total_predicted_targets'] = 6144
        elif mutation == 'duplicate_article': bad['cohorts']['short']['article_ids'][1] = bad['cohorts']['short']['article_ids'][0]
        elif mutation == 'cross_stratum_article': bad['cohorts']['long']['article_ids'][0] = bad['cohorts']['short']['article_ids'][0]
        elif mutation == 'missing_offset': bad['cohorts']['long']['selected_offsets'].pop()
        elif mutation == 'overlapping_windows': bad['cohorts']['short']['selected_offsets'][1] = bad['cohorts']['short']['selected_offsets'][0]
        else: bad['cohorts']['long']['selected_offsets'][0] = 0
        bad_manifests.append((mutation, bad))
    for name, bad in bad_manifests:
        try:
            batch_plan(bad)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid manifest accepted')
    source = Path(__file__)
    report = {'status': 'all10240_targets_sixteen_batches_and_five_seed_article_ownership_verified',
        'sources_sha256': {p.name: sha(p) for p in [source, source.with_name('holdout_batch_layout.py'), source.with_name('holdout_uncertainty.py')]},
        'manifest_sha256': sha(manifest_path), 'batch_plan': plan,
        'synthetic_targets_per_seed': 10240, 'synthetic_seeds': 5,
        'invalid_inputs_rejected': rejected,
        'scope': 'Metadata and synthetic loss coverage only. No reserved token arrays, predictions or model scores opened. Not a forward-shape proof, full execution protocol, eligibility decision or authorization to score.'}
    with (root / 'holdout_batch_layout_preflight.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'batch_plan'}))


if __name__ == '__main__':
    main()
