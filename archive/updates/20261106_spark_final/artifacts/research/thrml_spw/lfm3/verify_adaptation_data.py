"""Independently reconstruct every reserved adaptation window and exclusions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_data'
    p = json.loads((out / 'protocol.json').read_text())
    m = json.loads((out / 'manifest.json').read_text())
    assert p['source_sha256'] == sha(Path(__file__).with_name('prepare_adaptation_data.py'))
    assert sha(out / 'protocol.json') == m['protocol_sha256']
    assert sha(out / 'tokens.npz') == m['tokens_sha256']
    assert sha(root / 'language_subset/tokens.npz') == p['frozen_test_tokens_sha256']
    tokenizer = AutoTokenizer.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint', local_files_only=True, trust_remote_code=True)
    test = np.load(root / 'language_subset/tokens.npz')
    frozen_hashes = {token_hash(row) for row in test['tokens']}
    assert len(frozen_hashes) == m['unique_forbidden_test_contexts'] == 8
    arrays = np.load(out / 'tokens.npz')
    training_hashes = set()
    observed = []
    for split in ['train', 'validation']:
        s = m['splits'][split]
        parquet = Path(s['parquet_path'])
        assert sha(parquet) == s['parquet_sha256']
        cached = Path(hf_hub_download(p['dataset'], f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                      repo_type='dataset', revision=p['dataset_revision'], local_files_only=True))
        assert sha(cached) == s['parquet_sha256']
        rows = pq.read_table(parquet, columns=['text'])['text'].to_pylist()
        stream = np.asarray(tokenizer('\n'.join(v for v in rows if v and v.strip()), add_special_tokens=False)['input_ids'], dtype=np.int64)
        tokens, offsets = arrays[split + '_tokens'], arrays[split + '_offsets']
        count, width = (256, 257) if split == 'train' else (32, 65)
        assert tokens.shape == (count, width) and tokens.dtype == np.int64
        assert offsets.tolist() == s['selected_offsets']
        assert len(stream) == s['full_token_count']
        assert s['predicted_tokens'] == count * (width - 1)
        assert np.all(offsets % width == 0) and np.all(np.diff(offsets) >= width)
        exclusion = frozen_hashes if split == 'train' else frozen_hashes | training_hashes
        seen = set()
        for start, row in zip(offsets, tokens, strict=True):
            np.testing.assert_array_equal(row, stream[start:start + width])
            row_hashes = {token_hash(window) for window in np.lib.stride_tricks.sliding_window_view(row, 65)}
            assert not row_hashes & exclusion
            seen.update(row_hashes)
        # Independently replay the ordered candidate reservation using the
        # saved seed and direct sliding-window comparisons via hashes.
        selected, rejected = [], []
        for tile in np.random.default_rng(p['selection_seeds'][split]).permutation(len(stream) // width):
            start = int(tile) * width
            candidate = stream[start:start + width]
            overlap = any(token_hash(w) in exclusion for w in np.lib.stride_tricks.sliding_window_view(candidate, 65))
            if overlap:
                rejected.append(start)
            else:
                selected.append(start)
                if len(selected) == count:
                    break
        assert sorted(selected) == offsets.tolist()
        assert rejected == s['skipped_offsets_due_to_exact65_context_overlap']
        if split == 'train':
            training_hashes = seen
        observed.append({'split': split, 'windows': count, 'predicted_tokens': count * (width - 1),
                         'unique65_contexts': len(seen), 'parquet_sha256': s['parquet_sha256']})
    source = Path(hf_hub_download(p['dataset'], 'wikitext-2-raw-v1/test-00000-of-00001.parquet',
                  repo_type='dataset', revision=p['dataset_revision'], local_files_only=True))
    assert sha(source) == sha(Path('data/wikitext/test.parquet'))
    texts = pq.read_table(source, columns=['text'])['text'].to_pylist()
    test_stream = np.asarray(tokenizer('\n'.join(v for v in texts if v and v.strip()), add_special_tokens=False)['input_ids'], dtype=np.int64)
    for offset, row in zip(test['token_offsets'], test['tokens'], strict=True):
        np.testing.assert_array_equal(row, test_stream[offset:offset + 65])
    audit = {'status': 'all_adaptation_windows_split_identity_and_exact65_context_exclusions_verified',
             'source_sha256': sha(Path(__file__)), 'manifest_sha256': sha(out / 'manifest.json'),
             'splits': observed, 'pinned_test_parquet_matches_original_and_tokens_exact': True,
             'scope': 'No model scores or training performed. Separate official train/validation/test splits and exact65-token exclusions verified; not phrase-level or pretraining decontamination. Existing512-token test remains consumed exploratory evidence.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
