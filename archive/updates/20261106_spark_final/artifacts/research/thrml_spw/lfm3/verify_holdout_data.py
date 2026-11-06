"""Reconstruct article boundaries, unscored holdout windows and reservation RNG."""
import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(row):
    return hashlib.sha256(np.asarray(row, dtype='<i8').tobytes()).hexdigest()


def grams(row):
    return {token_hash(v) for v in np.lib.stride_tricks.sliding_window_view(row, 65)}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'holdout_data'
    p = json.loads((out / 'protocol.json').read_text())
    m = json.loads((out / 'manifest.json').read_text())
    assert sha(Path(__file__).with_name('prepare_holdout_data.py')) == p['source_sha256']
    assert sha(out / 'protocol.json') == m['protocol_sha256']
    assert sha(out / 'tokens.npz') == m['tokens_sha256']
    assert sha(Path(p['parquet_path'])) == p['parquet_sha256'] == '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint')
    for name, value in p['tokenizer_sha256'].items():
        assert sha(checkpoint / name) == value
    for name, value in p['prerequisites_sha256'].items():
        assert sha(root / name) == value
    texts = pq.read_table(p['parquet_path'], columns=['text'])['text'].to_pylist()
    stitched, headings, position = [], [], 0
    for index, s in enumerate(texts):
        if not s or not s.strip():
            continue
        atoms = re.findall(r'\S+', s)
        leading = sum(1 for _ in itertools.takewhile(lambda v: v == '=', atoms))
        trailing = sum(1 for _ in itertools.takewhile(lambda v: v == '=', reversed(atoms)))
        if len(atoms) >= 3 and leading == trailing == 1:
            headings.append((position, index, hashlib.sha256(s.encode()).hexdigest()))
        stitched.append(s); position += len(s) + 1
    text = '\n'.join(stitched)
    if headings[0][0] != 0:
        headings.insert(0, (0, None, None))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    stream = np.asarray(encoded['input_ids'], dtype=np.int64)
    offsets = np.asarray(encoded['offset_mapping'], dtype=np.int64)
    assert len(stream) == m['global_test_token_count']
    assert token_hash(stream) == m['global_test_token_sha256']
    articles = []
    for index, (start, row_index, heading_hash) in enumerate(headings):
        stop = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        # Independent direct membership scan, rather than producer binary search.
        inside = np.flatnonzero((offsets[:, 0] >= start) & (offsets[:, 1] <= stop))
        assert inside.size and np.all(np.diff(inside) == 1)
        articles.append({'article_id': index, 'char_start': start, 'char_end': stop,
            'heading_row': row_index, 'heading_sha256': heading_hash,
            'token_start': int(inside[0]), 'token_end': int(inside[-1]) + 1})
    assert articles == m['articles']
    legacy = np.load(root / 'language_subset/tokens.npz')
    excluded = set()
    for start, window in zip(legacy['token_offsets'], legacy['tokens'], strict=True):
        np.testing.assert_array_equal(stream[start:start + 65], window)
        lo, hi = offsets[start, 0], offsets[start + 64, 1]
        for article in articles:
            if article['char_start'] < hi and article['char_end'] > lo:
                excluded.add(article['article_id'])
    assert sorted(excluded) == m['excluded_consumed_LFM3_article_ids']
    adaptation = np.load(root / 'adaptation_data/tokens.npz')
    forbidden = set()
    for rows in [adaptation['train_tokens'], adaptation['validation_tokens'], legacy['tokens']]:
        for row in rows:
            forbidden |= grams(row)
    assert len(forbidden) == m['initial_forbidden65_context_count']
    cohort_arrays = np.load(out / 'tokens.npz')
    order = np.random.default_rng(p['article_seed']).permutation(len(articles))
    used = set()
    reports = []
    for label in p['selection_order']:
        c = p['cohorts'][label]
        width, required = c['window_length'], c['windows_per_article']
        expected, rejected = [], []
        for article_id in order:
            article_id = int(article_id)
            article = articles[article_id]
            if article_id in used or article_id in excluded or article['heading_row'] is None:
                continue
            count = (article['token_end'] - article['token_start']) // width
            permutation = np.random.default_rng(np.random.SeedSequence([p['window_seeds'][label], article_id])).permutation(count)
            pending, pending_hashes, rejected_offsets = [], set(), []
            for tile in permutation:
                start = article['token_start'] + int(tile) * width
                h = grams(stream[start:start + width])
                if any(value in forbidden or value in pending_hashes for value in h):
                    rejected_offsets.append(start)
                    continue
                pending.append(start); pending_hashes |= h
                if len(pending) == required:
                    break
            if len(pending) != required:
                rejected.append({'article_id': article_id, 'reason': 'insufficient_eligible_nonoverlapping_tiles',
                                 'candidate_tiles': count, 'excluded_offsets': rejected_offsets})
                continue
            used.add(article_id)
            forbidden |= pending_hashes
            expected.extend((start, article_id) for start in pending)
            if len(expected) == c['articles'] * required:
                break
        expected.sort()
        actual_offsets = cohort_arrays[label + '_offsets']
        actual_articles = cohort_arrays[label + '_article_ids']
        actual_tokens = cohort_arrays[label + '_tokens']
        assert actual_offsets.tolist() == [v[0] for v in expected] == m['cohorts'][label]['selected_offsets']
        assert actual_articles.tolist() == [v[1] for v in expected]
        assert actual_tokens.shape == (c['articles'] * required, width) and actual_tokens.dtype == np.int64
        assert np.all(np.diff(actual_offsets) >= width)
        assert sorted(set(actual_articles.tolist())) == m['cohorts'][label]['article_ids']
        assert rejected == m['cohorts'][label]['rejected_articles']
        for start, article_id, row in zip(actual_offsets, actual_articles, actual_tokens, strict=True):
            a = articles[article_id]
            assert a['token_start'] <= start and start + width <= a['token_end']
            np.testing.assert_array_equal(row, stream[start:start + width])
        for article_id in set(actual_articles):
            assert np.count_nonzero(actual_articles == article_id) == required
        assert not set(actual_articles) & excluded
        targets = len(actual_tokens) * (width - 1)
        assert targets == m['cohorts'][label]['predicted_tokens']
        reports.append({'cohort': label, 'articles': c['articles'], 'windows': len(actual_tokens),
                        'window_length': width, 'predicted_targets': targets})
    assert not set(cohort_arrays['short_article_ids']) & set(cohort_arrays['long_article_ids'])
    assert len(used) == 40 and sum(v['predicted_targets'] for v in reports) == m['total_predicted_targets'] == 10240
    audit = {'status': 'all_holdout_contexts_article_exclusions_and_reservations_verified',
        'source_sha256': sha(Path(__file__)), 'manifest_sha256': sha(out / 'manifest.json'),
        'cohorts': reports, 'distinct_articles': len(used), 'excluded_consumed_articles': len(excluded),
        'total_predicted_targets': 10240, 'global_test_token_sha256': token_hash(stream),
        'scope': 'Independent article-boundary membership, exact window/token reconstruction, all reserved train/validation/legacy65-context exclusions and seeded reservation replay. No model scores computed. Same corpus used in other-model studies; not pretraining/phrase decontamination or fullWikiTextbenchmark. Data-only reservation, no neural quality or equivalence evidence.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
