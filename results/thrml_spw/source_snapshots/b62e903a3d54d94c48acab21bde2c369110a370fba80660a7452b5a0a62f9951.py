"""Freeze separate language adaptation data; compute no LFM3 model scores."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def contexts(tokens, width=65):
    return {token_hash(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_data'
    out.mkdir(exist_ok=True)
    data = Path('data/thrml_spw/lfm3_1p7b/adaptation_parquet')
    data.mkdir(exist_ok=True)
    protocol = {'source_sha256': sha(Path(__file__)), 'dataset': 'Salesforce/wikitext',
                'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
                'configuration': 'wikitext-2-raw-v1',
                'model_revision': 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
                'frozen_test_tokens_sha256': sha(root / 'language_subset/tokens.npz'),
                'train_windows': 256, 'train_window_length': 257,
                'validation_windows': 32, 'validation_window_length': 65,
                'selection_seeds': {'train': 202610312, 'validation': 202610313},
                'selection': 'Seeded permutation of nonoverlapping token tiles in separate official splits; take first eligible count, then sort by offsets. Exclude any training example containing a complete65-token frozen test context; validation excludes every65-token context in selected training examples and frozen test. Raw newline-joined nonempty rows; no chat template or special tokens.',
                'scope': 'Preparation only; no LFM3 quality scores or training. Fixed train/validation subsets for future probability-aware adaptation and controls. Earlier different-model studies consumed WikiText validation/test. Exact65-token exclusion is not phrase-level or pretraining-corpus decontamination. The frozen512-token test remains exploratory, with earlier scores already consumed; not a new confirmatory test.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    tokenizer = AutoTokenizer.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
                                            local_files_only=True, trust_remote_code=True)
    frozen = np.load(root / 'language_subset/tokens.npz')['tokens']
    test_hashes = {token_hash(row) for row in frozen}
    forbidden = set(test_hashes)
    arrays, splits = {}, {}
    for split, count, length in [('train', 256, 257), ('validation', 32, 65)]:
        path = Path(hf_hub_download(protocol['dataset'],
                    f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type='dataset',
                    revision=protocol['dataset_revision'], token=False))
        local = data / (split + '.parquet')
        assert not local.exists()
        shutil.copyfile(path, local)
        assert sha(path) == sha(local)
        rows = pq.read_table(local, columns=['text'])['text'].to_pylist()
        ids = np.asarray(tokenizer('\n'.join(s for s in rows if s and s.strip()), add_special_tokens=False)['input_ids'], dtype=np.int64)
        order = np.random.default_rng(protocol['selection_seeds'][split]).permutation(len(ids) // length)
        selected, skipped = [], []
        for tile in order:
            offset = int(tile) * length
            window = ids[offset:offset + length]
            hashes = contexts(window)
            if hashes & forbidden:
                skipped.append(offset)
                continue
            selected.append(offset)
            if len(selected) == count:
                break
        assert len(selected) == count
        selected.sort()
        tokens = np.stack([ids[k:k + length] for k in selected])
        arrays[split + '_tokens'] = tokens
        arrays[split + '_offsets'] = np.asarray(selected, dtype=np.int64)
        if split == 'train':
            for row in tokens:
                forbidden.update(contexts(row))
        splits[split] = {'parquet_path': str(local), 'parquet_sha256': sha(local),
                         'full_token_count': len(ids), 'selected_offsets': selected,
                         'skipped_offsets_due_to_exact65_context_overlap': skipped,
                         'windows': count, 'window_length': length,
                         'predicted_tokens': count * (length - 1)}
    np.savez_compressed(out / 'tokens.npz', **arrays)
    manifest = {'protocol_sha256': sha(out / 'protocol.json'), 'tokens_sha256': sha(out / 'tokens.npz'),
                'splits': splits, 'unique_forbidden_test_contexts': len(test_hashes),
                'status': 'adaptation_inputs_frozen_no_training_or_model_scores'}
    with (out / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({'status': manifest['status'], 'train_predicted_tokens': 65536,
                      'validation_predicted_tokens': 2048, 'tokens_sha256': manifest['tokens_sha256']}))


if __name__ == '__main__':
    main()
