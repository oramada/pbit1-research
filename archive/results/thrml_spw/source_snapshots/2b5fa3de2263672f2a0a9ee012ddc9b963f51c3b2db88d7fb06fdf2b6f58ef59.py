"""Bit-pattern and corruption checks, including an existing validation excerpt."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .lossless_logits import write_logits, read_logits


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ['lossless_logits.py', 'preflight_lossless_logits.py']:
        path = Path(__file__).with_name(name)
        sources[name] = sha(path)
        assert (Path('results/thrml_spw/source_snapshots') / (sources[name] + '.py')).read_bytes() == path.read_bytes()
    all_high_words = (np.arange(65536, dtype='<u4') << 16).view('<f4').reshape(256, 256)
    rng = np.random.default_rng(202610650)
    random_bits = rng.integers(0, 2 ** 32, size=1048576, dtype=np.uint32).view(np.float32)
    boundary = np.array([0, 0x80000000, 1, 0x807fffff, 0x7f800000, 0xff800000,
                         0x7fc01234, 0x7f812345, 0xffffffff, 0x3f800001], dtype='<u4').view('<f4')
    cases = [('all_65536_high_words', all_high_words), ('random_full_words', random_bits),
             ('IEEE_boundaries_and_NaN_payloads', boundary), ('noncontiguous', all_high_words.T[::3, ::2]),
             ('empty', np.empty((0, 7), np.float32)), ('scalar', np.array(.12345, np.float32))]
    records = []
    for name, value in cases:
        path = args.output / (name + '.npz')
        metadata = write_logits(path, value)
        restored = read_logits(path)
        assert value.shape == restored.shape and value.dtype == restored.dtype
        assert value.tobytes(order='C') == restored.tobytes(order='C')
        records.append({'case': name, **metadata, 'file_sha256': sha(path)})
    assert records[0]['codec'] == 'exact_high16' and records[1]['codec'] == 'raw_float32'
    invalid_types = 0
    for value in [np.ones(2, np.float64), np.ones(2, np.int32)]:
        try:
            write_logits(args.output / 'invalid_type.npz', value)
        except ValueError:
            invalid_types += 1
        else:
            raise AssertionError('Unsupported type accepted')
    assert invalid_types == 2
    valid_path = args.output / 'all_65536_high_words.npz'
    with np.load(valid_path) as loaded:
        original = {k: loaded[k] for k in loaded.files}
    corruptions = 0
    for change in ['payload', 'same_size_shape', 'codec', 'version', 'digest', 'extra_field']:
        altered = {k: v.copy() for k, v in original.items()}
        if change == 'payload': altered['payload'][0] ^= np.uint16(1)
        elif change == 'same_size_shape': altered['shape'] = np.array([128, 512], dtype='<i8')
        elif change == 'codec': altered['codec'] = np.asarray(7, dtype=np.uint8)
        elif change == 'version': altered['format_version'] = np.asarray(2, dtype='<i8')
        elif change == 'digest': altered['data_sha256'][0] ^= np.uint8(1)
        else: altered['unrecognized'] = np.asarray(0)
        path = args.output / ('corrupt_' + change + '.npz')
        with path.open('xb') as handle: np.savez_compressed(handle, **altered)
        try:
            read_logits(path)
        except ValueError:
            corruptions += 1
        else:
            raise AssertionError('Corrupted archive accepted')
    root = Path('results/thrml_spw/lfm3_1p7b')
    result_path = root / 'counted_mean_t4096/mean/202610420/results.json'
    audit_path = result_path.with_name('independent_audit.json')
    result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
    assert audit['results_sha256'] == sha(result_path)
    raw_path = Path(result['outputs'][0]['path'])
    assert sha(raw_path) == result['outputs'][0]['sha256']
    with np.load(raw_path) as raw:
        excerpt = raw['native'][0, :4, :512].copy()
    path = args.output / 'existing_validation_excerpt.npz'
    metadata = write_logits(path, excerpt)
    assert read_logits(path).tobytes() == excerpt.tobytes()
    report = {'status': 'lossless_float32_bit_patterns_corruption_checks_and_validation_excerpt_verified',
              'sources_sha256': sources, 'synthetic_cases': records, 'invalid_types_rejected': invalid_types,
              'corrupted_archives_rejected': corruptions, 'existing_validation_excerpt': {
                  'original_raw_path': str(raw_path), 'original_raw_sha256': result['outputs'][0]['sha256'],
                  'selection': 'native logits window0, positions0:4, vocabulary0:512', **metadata,
                  'archive_sha256': sha(path)},
              'scope': 'Observation storage only, not weight encoding or inference precision. Every restored float32 bit and shape matches, including signed zeros/subnormals/NaN payloads. All65536high16patterns exhaustive; full32path tested on1048576randomwords/boundaries. Existing validation excerpt only; no holdout tokens or scores read.'}
    with (args.output / 'results.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
