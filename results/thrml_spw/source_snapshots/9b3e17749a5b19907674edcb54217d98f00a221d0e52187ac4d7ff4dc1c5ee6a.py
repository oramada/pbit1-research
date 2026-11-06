"""Independent full-checkpoint command/metadata fingerprints and anchor audit."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'probability_tensor_integration'
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(root / 'meta_inventory.json') == p['inventory_sha256']
    assert sha(root / 'preparation_inventory.json') == p['checkpoint_inventory_sha256']
    assert sha(root / 'baseline_smoke.npz') == p['baseline_input_sha256']
    meta = json.loads((root / 'meta_inventory.json').read_text())
    records = {v['canonical_parameter']: v for v in r['encoders']}
    assert set(records) == {v['name'] for v in meta['parameters']}
    assert len(records) == r['unique_encoders'] == 286
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    expected = json.loads((root / 'preparation_inventory.json').read_text())['checkpoint_files']['model.safetensors']
    assert sha(checkpoint) == expected
    total = groups = zero_groups = zero_weights = 0
    with safe_open(str(checkpoint), framework='pt') as f:
        for param in meta['parameters']:
            name, shape = param['name'], param['shape']
            rec = records[name]
            assert rec['shape'] == shape and rec['numel'] == math.prod(shape)
            assert rec['theta_dtype'] == rec['metadata_dtype'] == 'torch.float32'
            assert rec['surrogate_forward_uses'] >= 1
            vector = len(shape) == 1
            width = shape[0] if vector else math.prod(shape[1:])
            rows = 1 if vector else shape[0]
            size = min(128, width)
            assert width % size == 0 and rec['group_size'] == size
            assert rec['groups'] == param['numel'] // size
            hashes = {key: hashlib.sha256() for key in ['codes', 'lower', 'span']}
            for start in range(0, rows, 128):
                tensor = f.get_tensor(name) if vector else f.get_slice(name)[start:min(start + 128, rows)]
                w = tensor.float().numpy().astype(np.float64).reshape(-1, width // size, size)
                lo = w.min(2, keepdims=True)
                span = w.max(2, keepdims=True) - lo
                probability = np.divide(w - lo, span, out=np.full_like(w, .5), where=span != 0).clip(1e-12, 1 - 1e-12)
                field = ((np.log(probability) - np.log1p(-probability)) / 2).clip(-4, 4)
                codes = np.rint((field + 4) * 4095 / 8).astype('<u2')
                hashes['codes'].update(codes.tobytes())
                hashes['lower'].update(lo.astype('<f4').tobytes())
                hashes['span'].update(span.astype('<f4').tobytes())
                zero_groups += int(np.count_nonzero(span == 0))
                zero_weights += int(np.count_nonzero(span == 0)) * size
            for key, h in hashes.items():
                assert h.hexdigest() == rec[key + '_sha256'], (name, key)
            assert rec['codes_sha256'] == rec['initial_codes_sha256']
            total += param['numel']; groups += rec['groups']
            (out / 'audit_progress.json').write_text(json.dumps({'last_parameter': name, 'weights_verified': total}) + '\n')
    assert total == r['original_unique_learned_weights'] == r['trainable_probability_controls'] == r['frozen_original_reference_weights_retained'] == 1728801536
    assert groups == r['range_groups'] == 13531430
    names = {v['name'].removesuffix('.weight') for v in meta['parameters']} | {'lm_head'}
    assert set(r['module_aliases']) == names and len(names) == r['weighted_modules'] == 287
    assert r['module_aliases']['lm_head'] == r['module_aliases']['model.embed_tokens'] == 'model.embed_tokens.weight'
    for name, canonical in r['module_aliases'].items():
        assert canonical == ('model.embed_tokens.weight' if name == 'lm_head' else name + '.weight')
    assert r['embedding_head_encoder_shared'] and records['model.embed_tokens.weight']['surrogate_forward_uses'] >= 2
    anchors = json.loads((out / 'anchor_results.json').read_text())
    assert sha(out / 'anchors.npz') == anchors['arrays_sha256']
    arrays = np.load(out / 'anchors.npz')
    precision = json.loads((root / 'precision_preflight/results.json').read_text())
    for row, native_call in zip(anchors['anchors'], [25, 27, 29, 31], strict=True):
        kind = row['kind']
        assert sha(Path(row['input_path'])) == row['input_sha256']
        assert sha(Path(row['reference_path'])) == row['reference_sha256']
        a = np.load(row['input_path'])
        codes = arrays[kind + '_codes']
        nrows = codes.shape[0] if codes.ndim > 1 else 1
        width = codes.size // nrows
        size = min(128, width)
        decoded = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        q = np.exp(-np.logaddexp(0, -2 * decoded.astype(np.float64))).reshape(nrows, width // size, size)
        lo, span = arrays[kind + '_lower'].astype(np.float64), arrays[kind + '_span'].astype(np.float64)
        mean = (lo + span * q).reshape(codes.shape)
        variance = (span * span * q * (1 - q)).reshape(codes.shape)
        np.testing.assert_allclose(arrays[kind + '_mean'], mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(arrays[kind + '_variance'], variance, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(arrays[kind + '_output'], np.load(row['reference_path']), rtol=2e-12, atol=2e-12)
        worker_layout = codes.reshape(codes.shape[0], -1, size).transpose(1, 2, 0) if kind in ['linear', 'conv'] else codes.reshape(1, 1, -1)
        for tile, start in enumerate(range(0, worker_layout.shape[-1], 64)):
            path = root / 'precision_preflight/raw_states' / f'call{native_call}_tile{tile}.npz'
            assert sha(path) == precision['artifacts_sha256'][str(path.relative_to(root / 'precision_preflight'))]
            reference = np.load(path)['codes']
            length = min(64, worker_layout.shape[-1] - start)
            np.testing.assert_array_equal(worker_layout[:, :, start:start + length], reference[:, :, :length])
    assert anchors['shared_trainable_controls'] == 4096 and anchors['both_paths_contribute']
    ge, gh, combined = [arrays[k] for k in ['tied_embedding_gradient', 'tied_head_gradient', 'tied_combined_gradient']]
    assert np.count_nonzero(ge) > 0 and np.count_nonzero(gh) > 0
    np.testing.assert_allclose(combined, ge + gh, rtol=1e-6, atol=1e-6)
    assert sha(out / 'forward_arrays.npz') == r['forward_arrays_sha256']
    forward = np.load(out / 'forward_arrays.npz')
    np.testing.assert_array_equal(forward['input_ids'], np.load(root / 'baseline_smoke.npz')['input_ids'])
    baseline, surrogate = forward['CPU_BF16'], forward['CPU_mean_weight_surrogate']
    assert baseline.shape == surrogate.shape == (1, 16, 128000)
    assert np.isfinite(baseline).all() and np.isfinite(surrogate).all()
    np.testing.assert_allclose(r['CPU_mean_weight_logit_MSE_vs_CPU_BF16'], np.mean((surrogate.astype(np.float64) - baseline) ** 2), rtol=1e-12, atol=1e-12)
    assert r['CPU_mean_weight_argmax_agreement'] == float(np.mean(surrogate.argmax(-1) == baseline.argmax(-1)))
    audit = {'status': 'all_checkpoint_command_metadata_fingerprints_anchors_and_saved_forward_metrics_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'all_unique_controls': total, 'zero_span_groups': zero_groups, 'weights_in_zero_span_groups': zero_weights,
        'scope': 'Independent full-coordinate checkpoint command/metadata hashes and prior public-command anchors, tied-gradient addition and saved CPU forward metrics. Does not rerun the complete CPU model; forward-use counters are producer runtime evidence. Zero-span groups stay fixed with fixed metadata despite trainable control slots. No full-model backward, training, ACT12 full-model operator parity, native sampling, quality equivalence or hardware advantage claimed.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
