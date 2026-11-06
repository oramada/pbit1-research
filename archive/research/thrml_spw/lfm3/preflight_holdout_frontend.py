"""Synthetic variable-length geometry check with independent canonical operators."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .holdout_frontend import prepare, restore


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def program_for(shape, rng):
    width = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    rows = shape[0] if len(shape) > 1 else 1
    group = min(128, width)
    codes = rng.integers(0, 4096, shape, dtype=np.uint16)
    lower = rng.integers(-8, 9, (rows, width // group, 1)).astype(np.float32) / 16
    span = rng.integers(1, 9, lower.shape).astype(np.float32) / 16
    return {'codes': codes, 'lower': lower, 'span': span, 'shape': np.asarray(shape), 'group_size': np.asarray(group)}


def main():
    rng = np.random.default_rng(202610701)
    records = []
    for batch, length in [(8, 65), (4, 257)]:
        for kind, shape in [('linear', (129, 2048)), ('conv', (2048, 1, 7)),
                            ('elementwise', (2048,)), ('embedding', (33, 2048))]:
            p = program_for(shape, rng)
            input_shape = ((batch, 2048, length) if kind == 'conv' else
                           (batch, length) if kind == 'embedding' else (batch, length, 2048))
            x = (rng.integers(0, 33, input_shape, dtype=np.int64) if kind == 'embedding'
                 else rng.integers(-3, 4, input_shape).astype(np.float32) / 8)
            xx, layout, state = prepare(kind, x, p)
            # Synthetic exact dyadic probabilities avoid sigmoid and summation
            # tolerances obscuring the geometry. This is not device simulation.
            probability = p['codes'].astype(np.float64) / 4096
            rows = shape[0] if len(shape) > 1 else 1
            canonical = (p['lower'] + p['span'] * probability.reshape(rows, -1, int(p['group_size']))).reshape(shape)
            q = layout['codes'].astype(np.float64) / 4096
            size = min(128, q.shape[0])
            wm = (layout['lower'][:, None] + layout['span'][:, None] * q.reshape(-1, size, q.shape[1])).reshape(q.shape)
            actual_flat = (xx.astype(np.float64) @ wm if kind == 'linear' else
                           (xx.astype(np.float64) * wm[None]).sum(axis=1))
            actual = restore(actual_flat, state)
            if kind == 'linear':
                expected = x.astype(np.float64) @ canonical.T
            elif kind == 'elementwise':
                expected = x.astype(np.float64) * canonical
            elif kind == 'embedding':
                expected = canonical[x]
            else:
                # Explicit seven-tap causal convolution, independent of the
                # sliding-window transpose used by prepare(). Extra six output
                # positions match the original padding-six Conv1d hook contract.
                expected = np.zeros((batch, 2048, length + 6), dtype=np.float64)
                for tap in range(7):
                    offset = 6 - tap
                    expected[:, :, offset:length] += x[:, :, :length-offset].astype(np.float64) * canonical[:, 0, tap][None, :, None]
            np.testing.assert_array_equal(actual, expected)
            try:
                restore(actual_flat[:, :-1], state)
            except ValueError:
                pass
            else:
                raise AssertionError('Truncated worker output accepted')
            records.append({'kind': kind, 'input_shape': list(x.shape),
                'worker_input_shape': list(xx.shape), 'output_shape': list(actual.shape),
                'output_values': actual.size, 'canonical_output_bitwise_equal': True,
                'truncated_output_rejected': True})
    source = Path(__file__)
    report = {'status': 'eight_synthetic_short_long_frontend_canonical_operator_geometries_verified',
        'sources_sha256': {p.name: sha(p) for p in [source, source.with_name('holdout_frontend.py'), source.with_name('program_export.py')]},
        'cases': records,
        'scope': 'Synthetic shape/layout/restoration only; 2048-channel activations at8x65 and4x257. Dyadic synthetic probabilities are not physical marginals. No THRML sampling, ACT12 quantization, normalization, full-model attention, BF16 rounding, heldout token arrays or predictions tested. No adoption in any frozen active writer. Future complete scoring protocol remains required.'}
    with Path('results/thrml_spw/lfm3_1p7b/holdout_frontend_preflight.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
