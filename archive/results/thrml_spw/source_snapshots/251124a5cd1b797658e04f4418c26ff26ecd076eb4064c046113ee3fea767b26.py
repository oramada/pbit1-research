"""Independent canonical-command mapping and public-trajectory readout audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    out = Path('results/thrml_spw/lfm3_1p7b/program_export_preflight')
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for path, digest in r['artifacts_sha256'].items():
        assert sha(Path(path)) == digest
    assert r['status'] == 'all34_direct_program_cases_passed' and len(r['receipts']) == 34
    assert len(r['programs']) == 9
    for record in r['programs']:
        assert sha(Path(record['source_input'])) == record['source_input_sha256']
        program = np.load(record['program'])
        assert set(program.files) == {'codes', 'lower', 'span', 'shape', 'group_size', 'format_version'}
        assert int(program['format_version']) == 1 and program['codes'].dtype == np.uint16
        assert np.max(program['codes']) <= 4095
        assert program['lower'].dtype == program['span'].dtype == np.float32
        source = np.load(record['source_input'])
        kind, variant = record['kind'], record['variant']
        if variant == 'trained':
            canonical = source['after_codes'].reshape(128, 8).T
            lower = source['lower'].reshape(8, 1, 1).astype(np.float32)
            span = source['span'].reshape(8, 1, 1).astype(np.float32)
        else:
            weight = source['w'].T if kind == 'linear' else source['w'].T[:, None, :] if kind == 'conv' else source['w'].reshape(2, 2048) if kind == 'embedding' else source['w'].reshape(-1)
            rows = weight.shape[0] if weight.ndim > 1 else 1
            width = weight.size // rows
            size = min(128, width)
            w = weight.astype(np.float64).reshape(rows, -1, size)
            lo = w.min(2, keepdims=True)
            width_of_range = w.max(2, keepdims=True) - lo
            probability = np.divide(w - lo, width_of_range, out=np.full_like(w, .5), where=width_of_range != 0).clip(1e-12, 1 - 1e-12)
            ideal = ((np.log(probability) - np.log1p(-probability)) / 2).clip(-4, 4)
            initial = np.rint((ideal + 4) * 4095 / 8).astype(np.uint16)
            theta = (initial.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)).reshape(weight.shape)
            if variant == 'perturbed':
                # Reproduce the declared FP32 perturbation grid exactly;
                # command encoding itself is independently computed in NumPy.
                delta = torch.linspace(-.25, .25, theta.size).numpy().reshape(theta.shape)
                theta = np.clip(theta + delta, -4, 4)
            canonical = np.rint((theta.astype(np.float64) + 4) * 4095 / 8).astype(np.uint16)
            lower, span = lo.astype(np.float32), width_of_range.astype(np.float32)
        np.testing.assert_array_equal(program['codes'], canonical)
        np.testing.assert_array_equal(program['lower'], lower)
        np.testing.assert_array_equal(program['span'], span)
        assert program['shape'].tolist() == list(canonical.shape)
        size = int(program['group_size'])
        payload = np.load(record['input'])
        assert set(payload.files) == {'x', 'codes', 'lower', 'span'} and 'w' not in payload.files
        if kind in ['linear', 'conv']:
            expected_codes = canonical.T if kind == 'linear' else canonical[:, 0, :].T
            expected_lower, expected_span = lower[:, :, 0].T, span[:, :, 0].T
        else:
            expected_codes = canonical.reshape(1, -1)
            expected_lower = np.repeat(lower, size, axis=2).reshape(1, -1)
            expected_span = np.repeat(span, size, axis=2).reshape(1, -1)
        for key, expected in [('codes', expected_codes), ('lower', expected_lower), ('span', expected_span)]:
            np.testing.assert_array_equal(payload[key], expected)
    seen = set()
    expected_files = set()
    tile_count = counter_entries = observations = 0
    for row in r['receipts']:
        kind, mode, t = row['kind'], row['mode'], row['samples']
        seen.add((kind, row['variant'], mode, t))
        a = np.load(row['input'])
        codes, x = a['codes'], a['x']
        fanin, channels = codes.shape
        size = min(128, fanin)
        groups = fanin // size
        depthwise = kind != 'linear'
        scale = float(np.abs(x).max()) / 2047 if np.any(x) else 1.0
        integers = np.rint(x / scale).clip(-2047, 2047).astype(np.int64)
        outputs = []
        padding = 0
        for tile, start in enumerate(range(0, channels, 64)):
            width = min(64, channels - start)
            c = np.full((fanin, 64), 2048, np.uint16)
            c[:, :width] = codes[:, start:start + width]
            c = c.reshape(groups, size, 64)
            lo, span = np.zeros((groups, 64)), np.zeros((groups, 64))
            lo[:, :width] = a['lower'][:, start:start + width]
            span[:, :width] = a['span'][:, start:start + width]
            xt = integers[:, :, start:start + width] if depthwise else integers
            if depthwise and width < 64:
                xt = np.pad(xt, ((0, 0), (0, 0), (0, 64 - width)))
            xg = xt.reshape(xt.shape[0], groups, size, 64) if depthwise else xt.reshape(xt.shape[0], groups, size)
            if mode == 'native':
                path = out / 'raw_states' / f"call{row['call']}_tile{tile}.npz"
                expected_files.add(str(path))
                raw = np.load(path)
                np.testing.assert_array_equal(raw['codes'], c)
                np.testing.assert_array_equal(raw['inputs'], xt)
                states = raw['states']
                assert states.shape == (t, groups, size, 64) and states.dtype == np.bool_
                counts = states.sum(0, dtype=np.int64)
                total = np.sum(xg * counts[None], axis=2, dtype=np.int64) if depthwise else np.einsum('bgk,gko->bgo', xg, counts)
                np.testing.assert_array_equal(total, raw['total'].astype(np.int64))
                weighted = total.astype(np.float64) / t
                tile_count += 1; counter_entries += total.size; observations += states.size
            else:
                fields = c.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
                q = np.exp(-np.logaddexp(0, -2 * fields.astype(np.float64)))
                weighted = np.sum(xg * q[None], axis=2) if depthwise else np.einsum('bgk,gko->bgo', xg, q)
            offset = xg.sum(2) if depthwise else xg.sum(2)[:, :, None]
            outputs.append(((weighted * span[None] + offset * lo[None]).sum(1) * scale)[:, :width])
            padding += fanin * (64 - width)
        np.testing.assert_allclose(np.concatenate(outputs, axis=1), np.load(row['output']), rtol=2e-12, atol=2e-12)
        assert row['program_codes_used_directly'] and not row['reference_weights_supplied']
        assert row['weight_coordinates'] == codes.size and row['padding_nodes'] == padding
        assert row['retained_evaluations'] == ((codes.size + padding) * t if mode == 'native' else 0)
        assert row['warmup_evaluations'] == (codes.size + padding if mode == 'native' else 0)
    expected_cases = {(k, v, mode, t) for k in ['linear', 'conv', 'embedding', 'elementwise']
                      for v in ['initial', 'perturbed'] for mode in ['mean', 'native'] for t in [1, 16]}
    expected_cases |= {('linear', 'trained', mode, 16) for mode in ['mean', 'native']}
    assert seen == expected_cases
    assert {str(v) for v in (out / 'raw_states').glob('*.npz')} == expected_files
    audit = {'status': 'all34_direct_command_serialization_mapping_public_counters_and_readouts_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'programs': 9, 'public_trajectory_tiles': tile_count,
             'exact_integer_accumulator_entries': counter_entries, 'public_binary_observations': observations,
             'scope': 'Four tensor kinds plus actual trained1024-weight projectionpilot, preserved fixed metadata, no reference-weight input or range re-encoding. All cases retained. CPU official THRML trajectories independently replayed offline. No full-model trained export, language quality, BF16 equivalence or hardware cost benefit claimed.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
