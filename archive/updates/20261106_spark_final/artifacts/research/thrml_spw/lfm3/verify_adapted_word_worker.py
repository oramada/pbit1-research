"""Independent audit of tiled word requests, every raw trajectory and work count."""
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
    args = parser.parse_args()
    source = Path(__file__)
    assert sha(Path('results/thrml_spw/source_snapshots') / (sha(source) + '.py')) == sha(source)
    result = json.loads((args.input / 'results.json').read_text())
    protocol = json.loads((args.input / 'protocol.json').read_text())
    assert sha(args.input / 'protocol.json') == result['protocol_sha256']
    for name, digest in protocol['sources_sha256'].items():
        assert sha(source.with_name(name)) == digest
    assert len(result['records']) == 16
    public_bits = 0
    max_error = 0.0
    for ordinal, rec in enumerate(result['records']):
        request, receipt = rec['request'], rec['receipt']
        kind, fanin, channels, bits, samples, batch = protocol['cases'][ordinal // 2]
        mode = 'native' if ordinal % 2 else 'mean'
        assert request['kind'] == kind and request['mode'] == mode and request['call'] == ordinal
        assert request['samples'] == samples and request['word_bits'] == bits
        assert request['key'] == 202610440 + ordinal // 2
        input_path, output_path = Path(request['input']), Path(request['output'])
        assert sha(input_path) == rec['input_sha256'] and sha(output_path) == rec['output_sha256']
        with np.load(input_path) as loaded:
            xf, words, lower, span = [loaded[k] for k in ['x', 'words', 'lower', 'span']]
        size = min(128, fanin)
        groups, tiles = fanin // size, (channels + 63) // 64
        assert words.shape == (fanin, channels) and words.dtype == np.uint8
        assert words.max() < 2 ** bits and lower.shape == span.shape == (groups, channels)
        assert lower.dtype == span.dtype == np.float32
        scale = float(np.max(np.abs(xf))) / 2047 if np.any(xf) else 1.0
        x = np.rint(xf / scale).clip(-2047, 2047).astype(np.int32)
        if mode == 'native':
            assert len(rec['raw_traces']) == tiles * bits
        else:
            assert len(rec['raw_traces']) == 0
        trace_paths = {}
        for trace in rec['raw_traces']:
            path = Path(trace['path'])
            assert sha(path) == trace['sha256']
            assert path.parent.resolve() == Path(request['state_dir']).resolve()
            trace_paths[path.name] = path
        pieces = []
        for tile in range(tiles):
            first = 64 * tile
            width = min(64, channels - first)
            w = words[:, first:first + width].reshape(groups, size, width)
            lo, sp = lower[:, first:first + width].astype(np.float64), span[:, first:first + width].astype(np.float64)
            w = np.pad(w, ((0, 0), (0, 0), (0, 64 - width)))
            lo, sp = np.pad(lo, ((0, 0), (0, 64 - width))), np.pad(sp, ((0, 0), (0, 64 - width)))
            depthwise = kind != 'linear'
            xt = x[:, :, first:first + width] if depthwise else x
            if depthwise:
                xt = np.pad(xt, ((0, 0), (0, 0), (0, 64 - width)))
            xg = xt.reshape(batch, groups, size, 64) if depthwise else xt.reshape(batch, groups, size)
            if mode == 'native':
                weighted_counts = np.zeros(w.shape, np.int64)
                for bit in range(bits):
                    path = trace_paths[f'call{ordinal}_tile{tile}_bit{bit}.npz']
                    with np.load(path) as raw:
                        states, counts, total = raw['states'], raw['counts'], raw['total']
                        assert states.dtype == np.bool_ and states.shape == (samples, groups, size, 64)
                        for name, expected in [('words', w), ('lower', lo), ('span', sp), ('inputs', xt)]:
                            np.testing.assert_array_equal(raw[name], expected)
                        assert float(raw['scale']) == scale and int(raw['width']) == width
                    c = states.sum(axis=0, dtype=np.int64)
                    np.testing.assert_array_equal(c, counts)
                    expression = 'bgko,tgko->bgo' if depthwise else 'bgk,tgko->bgo'
                    contraction = np.einsum(expression, xg.astype(np.int64), states.astype(np.int64), dtype=np.int64, optimize=False)
                    np.testing.assert_array_equal(contraction, total)
                    weighted_counts += c * (1 << bit)
                    public_bits += states.size
                probability = weighted_counts.astype(np.float64) / (((1 << bits) - 1) * samples)
            else:
                p0, p1 = 1 / (1 + np.exp(8.0)), 1 / (1 + np.exp(-8.0))
                probability = p0 + (p1 - p0) * w.astype(np.float64) / ((1 << bits) - 1)
            mean_weight = lo[:, None, :] + sp[:, None, :] * probability
            expression = 'bgko,gko->bo' if depthwise else 'bgk,gko->bo'
            expected = np.einsum(expression, xg.astype(np.float64), mean_weight) * scale
            pieces.append(expected[:, :width])
        expected = np.concatenate(pieces, axis=1)
        actual = np.load(output_path)
        assert actual.shape == (batch, channels) and np.isfinite(actual).all()
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
        padding = fanin * (tiles * 64 - channels)
        assert receipt['weight_coordinates'] == fanin * channels
        assert receipt['padding_parameter_slots'] == padding and receipt['padding_nodes'] == bits * padding
        assert receipt['retained_evaluations'] == (bits * (fanin * channels + padding) * samples if mode == 'native' else 0)
        assert receipt['warmup_evaluations'] == (bits * (fanin * channels + padding) if mode == 'native' else 0)
        assert receipt['simulation_count_carry_bytes'] == (4 * min(4, tiles) * fanin * 64 if mode == 'native' else 0)
        assert receipt['max_simultaneous_sampler_state_slots'] == min(4, tiles) * fanin * 64
        assert receipt['program_words_used_directly'] and not receipt['reference_weights_supplied']
    audit = {'status': 'all_16_CPU_word_requests_raw_trajectories_quantization_tiling_and_work_verified',
             'source_sha256': sha(source), 'results_sha256': sha(args.input / 'results.json'),
             'raw_public_bits_checked': public_bits, 'requests_checked': 16,
             'maximum_independently_reordered_readout_error': max_error, 'readout_absolute_tolerance': 1e-9,
             'scope': 'All raw CPU trajectories and public-state int64 sums, finite-word mean readouts, ACT12 scaling, 64-column tails, four-lane chunking, zero ranges/inputs, embedding/norm/conv/linear layouts and actual7680fanin. No native GPU or full-model quality/hardware claim.'}
    with (args.input / 'independent_audit.json').open('x') as handle:
        json.dump(audit, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
