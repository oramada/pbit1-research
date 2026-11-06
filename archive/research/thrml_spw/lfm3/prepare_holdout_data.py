"""Reserve unscored, article-separated LFM3 holdout contexts at two lengths."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pyarrow.parquet as pq
import transformers
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def contexts(tokens):
    return {digest(tokens[i:i + 65]) for i in range(len(tokens) - 64)}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'holdout_data'
    out.mkdir(exist_ok=True)
    source = Path('data/wikitext/test.parquet')
    assert sha(source) == '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint')
    protocol = {'source_sha256': sha(Path(__file__)), 'dataset': 'Salesforce/wikitext',
        'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3', 'configuration': 'wikitext-2-raw-v1',
        'parquet_path': str(source), 'parquet_sha256': sha(source),
        'model_revision': 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'tokenizer_sha256': {name: sha(checkpoint / name) for name in
            ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json']},
        'prerequisites_sha256': {name: sha(root / name) for name in
            ['adaptation_data/independent_audit.json', 'adaptation_data/tokens.npz', 'language_subset/tokens.npz', 'language_subset/manifest.json']},
        'article_seed': 202610360, 'window_seeds': {'short': 202610361, 'long': 202610362},
        'cohorts': {'long': {'articles': 16, 'windows_per_article': 1, 'window_length': 257},
                    'short': {'articles': 24, 'windows_per_article': 4, 'window_length': 65}},
        'selection_order': ['long', 'short'],
        'selection': 'Tokenize the identical global newline-joined nonempty test stream with offsets, no special/chat tokens. Primary article headings have exactly one leading/trailing standalone equals token. Exclude every article intersecting any of the eight consumed LFM3 test windows. Seeded permutation of remaining article IDs; for each cohort take first eligible article groups. Within each article, seed PCG64 with SeedSequence([cohort_window_seed, article_id]); permute nonoverlapping tiles beginning at the first wholly contained token. Select required first eligible windows, then sort global offsets. Long and short cohorts have disjoint articles. Exclude any selected window containing a65-token context from ALL256reserved257-token train windows,32reserved65-token validation windows,8consumed65-token test windows or previously reserved holdout windows. Reject an article if not enough eligible tiles; failed partial reservations do not affect later selection.',
        'scope': 'Preparation only, no model forward or quality score. New LFM3 holdout contexts exclude consumed LFM3 test ARTICLES and exact65-token overlaps with all reserved train/validation data. Same test corpus previously used in other-model experiments, and pretrained-corpus/article/phrase decontamination is not claimed. Article-balanced sample at two lengths, not uniform full-WikiText perplexity. Scores must stay unconsumed until a separate treatment/evaluation protocol is fixed; this reservation itself does not authorize a confirmatory equivalence claim.',
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'transformers': transformers.__version__}}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    assert json.loads((root / 'adaptation_data/independent_audit.json').read_text())['status'] == 'all_adaptation_windows_split_identity_and_exact65_context_exclusions_verified'
    rows = pq.read_table(source, columns=['text'])['text'].to_pylist()
    valid, headings = [], []
    cursor = 0
    for row_index, text in enumerate(rows):
        if not text or not text.strip():
            continue
        tokens = text.strip().split()
        title = len(tokens) > 2 and tokens[0] == tokens[-1] == '=' and tokens[1] != '=' and tokens[-2] != '='
        if title:
            headings.append({'char_start': cursor, 'heading_row': row_index, 'heading_sha256': hashlib.sha256(text.encode()).hexdigest()})
        valid.append(text)
        cursor += len(text) + 1
    text = '\n'.join(valid)
    if headings[0]['char_start'] != 0:
        headings.insert(0, {'char_start': 0, 'heading_row': None, 'heading_sha256': None})
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    assert tokenizer.is_fast
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    stream = np.asarray(encoded['input_ids'], dtype=np.int64)
    positions = np.asarray(encoded['offset_mapping'], dtype=np.int64)
    assert positions.shape == (len(stream), 2)
    assert np.all(np.diff(positions[:, 0]) >= 0) and np.all(np.diff(positions[:, 1]) >= 0)
    legacy = np.load(root / 'language_subset/tokens.npz')
    assert len(stream) == json.loads((root / 'language_subset/manifest.json').read_text())['token_count']
    for offset, window in zip(legacy['token_offsets'], legacy['tokens'], strict=True):
        np.testing.assert_array_equal(stream[offset:offset + 65], window)
    articles = []
    for index, heading in enumerate(headings):
        end = headings[index + 1]['char_start'] if index + 1 < len(headings) else len(text)
        lo = int(np.searchsorted(positions[:, 0], heading['char_start'], side='left'))
        hi = int(np.searchsorted(positions[:, 1], end, side='right'))
        articles.append({'article_id': index, **heading, 'char_end': end, 'token_start': lo, 'token_end': hi})
    consumed = set()
    for offset in legacy['token_offsets']:
        left, right = positions[offset, 0], positions[offset + 64, 1]
        consumed.update(a['article_id'] for a in articles if max(a['char_start'], left) < min(a['char_end'], right))
    adaptation = np.load(root / 'adaptation_data/tokens.npz')
    forbidden = set()
    for row in [*adaptation['train_tokens'], *adaptation['validation_tokens'], *legacy['tokens']]:
        forbidden.update(contexts(row))
    initial_forbidden_count = len(forbidden)
    order = np.random.default_rng(protocol['article_seed']).permutation(len(articles)).tolist()
    selected_articles, cohorts, arrays = set(), {}, {}
    for label in protocol['selection_order']:
        cfg = protocol['cohorts'][label]
        width, per_article = cfg['window_length'], cfg['windows_per_article']
        selected, rejected_articles = [], []
        for article_id in order:
            a = articles[article_id]
            if article_id in consumed or article_id in selected_articles or a['heading_row'] is None:
                continue
            candidates = (a['token_end'] - a['token_start']) // width
            rng = np.random.default_rng(np.random.SeedSequence([protocol['window_seeds'][label], article_id]))
            local, local_contexts, rejected_offsets = [], set(), []
            for tile in rng.permutation(candidates):
                offset = a['token_start'] + int(tile) * width
                hashes = contexts(stream[offset:offset + width])
                if hashes & (forbidden | local_contexts):
                    rejected_offsets.append(offset)
                    continue
                local.append(offset); local_contexts.update(hashes)
                if len(local) == per_article:
                    break
            if len(local) != per_article:
                rejected_articles.append({'article_id': article_id, 'reason': 'insufficient_eligible_nonoverlapping_tiles', 'candidate_tiles': candidates, 'excluded_offsets': rejected_offsets})
                continue
            selected_articles.add(article_id)
            forbidden.update(local_contexts)
            selected.extend((offset, article_id) for offset in local)
            if len(selected) == cfg['articles'] * per_article:
                break
        assert len(selected) == cfg['articles'] * per_article, (label, len(selected))
        selected.sort()
        offsets = np.asarray([v[0] for v in selected], dtype=np.int64)
        article_ids = np.asarray([v[1] for v in selected], dtype=np.int64)
        tokens = np.stack([stream[k:k + width] for k in offsets])
        arrays.update({label + '_tokens': tokens, label + '_offsets': offsets, label + '_article_ids': article_ids})
        cohorts[label] = {'windows': len(tokens), 'window_length': width, 'predicted_tokens': len(tokens) * (width - 1),
                          'article_ids': sorted(set(article_ids.tolist())), 'selected_offsets': offsets.tolist(), 'rejected_articles': rejected_articles}
    path = out / 'tokens.npz'
    np.savez_compressed(path, **arrays)
    manifest = {'status': 'article_separated_holdout_frozen_no_model_scores',
        'protocol_sha256': sha(out / 'protocol.json'), 'tokens_sha256': sha(path),
        'global_test_token_count': len(stream), 'global_test_token_sha256': digest(stream),
        'articles': articles, 'excluded_consumed_LFM3_article_ids': sorted(consumed),
        'initial_forbidden65_context_count': initial_forbidden_count, 'cohorts': cohorts,
        'total_predicted_targets': sum(v['predicted_tokens'] for v in cohorts.values())}
    with (out / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({'status': manifest['status'], 'targets': manifest['total_predicted_targets'],
                      'articles': len(selected_articles), 'excluded_consumed_articles': len(consumed),
                      'cohorts': {k: {'windows': v['windows'], 'targets': v['predicted_tokens']} for k, v in cohorts.items()}}))


if __name__ == '__main__':
    main()
