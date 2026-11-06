"""Audit preserved positional-word trajectories without importing JAX/THRML."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__)
    assert sha(Path('results/thrml_spw/source_snapshots') / (sha(source) + '.py')) == sha(source)
    report = json.loads(args.input.read_text())
    for name, digest in report['sources_sha256'].items():
        assert sha(source.with_name(name)) == digest
    assert len(report['cases']) == 6
    total_bits = 0
    maximum_readout_error = 0.0
    for record in report['cases']:
        path = Path(record['raw_path'])
        assert sha(path) == record['raw_sha256']
        with np.load(path) as raw:
            x, words, lo, span, y, states, counts, totals = [raw[k] for k in
                ['inputs', 'words', 'lower', 'span', 'output', 'states', 'counts', 'totals']]
        bits, fanin, samples = record['bits'], record['fanin'], record['samples']
        size = min(128, fanin)
        groups = fanin // size
        depthwise = record['depthwise']
        assert words.shape == (fanin, 64) and words.dtype == np.uint8
        assert np.max(words) < 2 ** bits
        assert states.dtype == np.bool_ and states.shape == (bits, samples, fanin * 64)
        assert counts.dtype == np.int32 and counts.shape == (bits, fanin * 64)
        assert totals.dtype == np.int32 and totals.shape == (bits, 3, groups, 64)
        assert x.dtype == np.int32 and np.max(np.abs(x.astype(np.int64))) <= 2047
        assert lo.shape == span.shape == (groups, 64)
        assert 2047 * size * samples < 2 ** 31
        xg = x.reshape(3, groups, size, 64) if depthwise else x.reshape(3, groups, size)
        weighted_counts = np.zeros((groups, size, 64), np.int64)
        for bit in range(bits):
            z = states[bit].reshape(samples, groups, size, 64).astype(np.int64)
            c = z.sum(axis=0, dtype=np.int64)
            np.testing.assert_array_equal(c.reshape(-1), counts[bit])
            expression = 'bgko,tgko->bgo' if depthwise else 'bgk,tgko->bgo'
            independent = np.einsum(expression, xg.astype(np.int64), z, dtype=np.int64, optimize=False)
            np.testing.assert_array_equal(independent, totals[bit])
            weighted_counts += c * (1 << bit)
        # A differently ordered readout: reconstruct temporal mean weights first.
        mean_weights = lo[:, None, :] + span[:, None, :] * (weighted_counts.astype(np.float64) / (((1 << bits) - 1) * samples))
        if depthwise:
            expected = np.einsum('bgko,gko->bo', xg.astype(np.float64), mean_weights)
        else:
            expected = np.einsum('bgk,gko->bo', xg.astype(np.float64), mean_weights)
        error = float(np.max(np.abs(expected - y)))
        maximum_readout_error = max(maximum_readout_error, error)
        np.testing.assert_allclose(y, expected, rtol=0, atol=1e-8)
        assert hashlib.sha256(y.tobytes()).hexdigest() == record['output_sha256']
        assert states.size == record['public_bits_checked'] == record['retained_scheduled_evaluations']
        assert record['warmup_scheduled_evaluations'] == bits * fanin * 64
        assert record['max_temporary_weight_count_bytes'] == 4 * fanin * 64
        total_bits += states.size
    assert total_bits == report['public_bits_checked']
    result = {'status': 'all_saved_native_word_trajectories_counts_int64_contractions_and_independent_readout_verified',
              'source_sha256': sha(source), 'report_sha256': sha(args.input),
              'cases': len(report['cases']), 'public_bits_checked': total_bits,
              'readout_absolute_tolerance': 1e-8, 'maximum_independently_reordered_readout_error': maximum_readout_error,
              'scope': 'All raw saved CPU trajectories audited without JAX/THRML import. Integer trajectory contractions and counts exact; independently reordered temporal mean-weight readout within floating tolerance. This does not prove task quality, native GPU execution, frontend tiling, bias calibration, correlation robustness or hardware efficiency.'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
