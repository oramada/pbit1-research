"""Offline eight-node field/sign/count/accumulator/readout verification."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/word8_preflight')
    protocol = json.loads((out / 'protocol.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    for name, key in [('preflight_word8.py', 'source_sha256'), ('word8_worker.py', 'worker_sha256'),
                      ('precision_worker.py', 'observer_sha256')]:
        assert sha(Path(__file__).with_name(name)) == protocol[key]
    for path, digest in {**protocol['inputs_sha256'], **results['artifacts_sha256']}.items():
        assert sha(Path(path)) == digest
    assert results['worker_ready']['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in results['worker_ready']['devices'])
    assert results['status'] == 'all16_operator_cases_passed' and len(results['receipts']) == 16
    tiles = entries = sign_coordinates = 0
    expected_raw = set()
    cases = set()
    for receipt in results['receipts']:
        kind, mode, t = receipt['kind'], receipt['mode'], receipt['samples']
        cases.add((kind, mode, t))
        a = np.load(receipt['input'])
        w, x = a['w'].astype(np.float64), a['x']
        depthwise = kind != 'linear'
        scale = np.max(np.abs(x)).item() / 2047 if np.any(x) else 1.0
        xint = np.rint(x / scale).clip(-2047, 2047).astype(np.int64)
        fanin, channels = w.shape
        size = min(128, fanin)
        groups = fanin // size
        reconstructed_outputs = []
        padding = 0
        for tile, start in enumerate(range(0, channels, 64)):
            width = min(64, channels - start)
            wt = np.zeros((fanin, 64), np.float64)
            wt[:, :width] = w[:, start:start + width]
            padding += fanin * (64 - width)
            grouped = wt.reshape(groups, size, 64)
            lower, span = grouped.min(1), grouped.max(1) - grouped.min(1)
            if kind in ['embedding', 'elementwise']:
                lower = np.zeros((1, 64), np.float64)
                span = np.zeros((1, 64), np.float64)
                lower[0, :width] = a['lower'][start:start + width]
                span[0, :width] = a['span'][start:start + width]
            normalized = np.divide(grouped - lower[:, None, :], span[:, None, :],
                                   out=np.full_like(grouped, .5), where=span[:, None, :] != 0).clip(0, 1)
            words = np.rint(255 * normalized).astype(np.uint8)
            xt = xint[:, :, start:start + width] if depthwise else xint
            if depthwise and width < 64:
                xt = np.pad(xt, ((0, 0), (0, 0), (0, 64 - width)))
            xg = xt.reshape(xt.shape[0], groups, size, 64) if depthwise else xt.reshape(xt.shape[0], groups, size)
            weighted = np.zeros((xt.shape[0], groups, 64), np.float64)
            for bit in range(8):
                sign = (words >> bit) & 1
                if mode == 'native':
                    path = out / 'raw_states' / f"call{receipt['call']}_tile{tile}_bit{bit}.npz"
                    expected_raw.add(str(path))
                    raw = np.load(path)
                    np.testing.assert_array_equal(raw['signs'], sign)
                    np.testing.assert_array_equal(raw['inputs'], xt)
                    states = raw['states']
                    assert states.shape == (t, groups, size, 64) and states.dtype == np.bool_
                    counts = states.sum(0, dtype=np.int64)
                    total = np.sum(xg * counts[None, :, :, :], axis=2, dtype=np.int64) if depthwise else np.einsum('bgk,gko->bgo', xg, counts)
                    np.testing.assert_array_equal(total, raw['total'].astype(np.int64))
                    weighted += total * (2 ** bit / (255 * t))
                    tiles += 1
                    entries += total.size
                    sign_coordinates += sign.size
                else:
                    q = np.exp(-np.logaddexp(0.0, np.where(sign, -8.0, 8.0)))
                    total = np.sum(xg * q[None, :, :, :], axis=2) if depthwise else np.einsum('bgk,gko->bgo', xg, q)
                    weighted += total * (2 ** bit / 255)
            offset = xg.sum(2) if depthwise else xg.sum(2)[:, :, None]
            y = ((weighted * span[None, :, :] + offset * lower[None, :, :]).sum(1) * scale)[:, :width]
            reconstructed_outputs.append(y)
        np.testing.assert_allclose(np.concatenate(reconstructed_outputs, axis=1), np.load(receipt['output']), rtol=2e-12, atol=2e-12)
        assert receipt['weight_coordinates'] == w.size
        assert receipt['logical_pbit_uses'] == 8 * w.size
        assert receipt['padding_parameter_slots'] == padding
        assert receipt['padding_nodes'] == padding * 8
        assert receipt['programmable_field_levels_per_node'] == 2 and receipt['field_bound'] == 4
        assert receipt['retained_evaluations'] == (8 * (w.size + padding) * t if mode == 'native' else 0)
        assert receipt['warmup_evaluations'] == (8 * (w.size + padding) if mode == 'native' else 0)
    assert cases == {(kind, mode, t) for kind in ['linear', 'conv', 'embedding', 'elementwise']
                     for mode in ['mean', 'native'] for t in [1, 16]}
    assert {str(p) for p in (out / 'raw_states').glob('*.npz')} == expected_raw
    audit = {'status': 'all16_cases_signs_public_counts_integer_accumulators_and_readouts_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'public_bit_trajectory_tiles': tiles, 'exact_integer_accumulator_entries': entries,
             'programmed_sign_coordinates_including_padding': sign_coordinates,
             'scope': 'Eight-node internal positional baseline validation on saved LFM3 operators, GPU THRML. Does not establish full language quality, equal-memory/energy performance or Extropic hardware encoding.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
