"""Stream independent first-step Adam replay from checkpoint and raw gradients.

Does not call the producer encoder, optimizer, serializer or program loader.
Checks saved outputs, not a second full-model forward/backward execution.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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
    out = root / 'optimizer_export_probe'
    protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert protocol['optimizer_steps'] == result['optimizer_steps'] == 1
    assert protocol['optimizer'] == 'Adam' and protocol['foreach'] is False
    assert protocol['learning_rate'] == .003 and protocol['betas'] == [.9, .999]
    assert protocol['epsilon'] == 1e-8 and protocol['weight_decay'] == 0
    for path, key in [('recomputed_probability_preflight/independent_audit.json', 'component_audit_sha256'),
                      ('adaptation_data/independent_audit.json', 'data_audit_sha256'),
                      ('adaptation_data/tokens.npz', 'train_tokens_sha256')]:
        assert sha(root / path) == protocol[key]
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    inventory = json.loads((root / 'preparation_inventory.json').read_text())
    assert sha(checkpoint) == inventory['checkpoint_files']['model.safetensors']
    meta = json.loads((root / 'meta_inventory.json').read_text())
    expected = {v['name']: v for v in meta['parameters']}
    programs = {v['name']: v for v in result['program_records']}
    gradients = {v['name']: v for v in result['gradient_records']}
    assert len(programs) == len(gradients) == 286
    assert set(programs) == set(gradients) == set(expected)
    total = groups = changed = nonzero = changed_programs = 0
    squared_norm = 0.
    with safe_open(str(checkpoint), framework='pt') as checkpoint_file:
        for name, param in expected.items():
            rec, gradrec = programs[name], gradients[name]
            shape = param['shape']
            assert rec['shape'] == gradrec['shape'] == shape
            assert rec['numel'] == gradrec['numel'] == math.prod(shape)
            program_path, gradient_path = Path(rec['path']), Path(gradrec['raw_gradient_path'])
            assert sha(program_path) == rec['file_sha256']
            assert sha(gradient_path) == gradrec['raw_gradient_file_sha256']
            gradient = np.load(gradient_path, mmap_mode='r')
            assert list(gradient.shape) == shape and gradient.dtype == np.float32
            with np.load(program_path) as saved:
                assert set(saved.files) == {'codes', 'lower', 'span', 'shape', 'group_size', 'format_version'}
                assert saved['shape'].tolist() == shape and int(saved['format_version']) == 1
                codes, lower, span = [saved[k] for k in ['codes', 'lower', 'span']]
                size_saved = int(saved['group_size'])
            assert codes.shape == gradient.shape and codes.dtype == np.uint16
            assert lower.dtype == span.dtype == np.float32
            vector = len(shape) == 1
            rows = 1 if vector else shape[0]
            width = shape[0] if vector else math.prod(shape[1:])
            size = min(128, width)
            assert size_saved == size and width % size == 0
            assert lower.shape == span.shape == (rows, width // size, 1)
            assert rec['groups'] == lower.size
            hashes = {key: hashlib.sha256() for key in ['initial_codes', 'codes', 'lower', 'span', 'theta_after', 'gradient']}
            parameter_changed = parameter_nonzero = 0
            parameter_squared_norm = 0.
            for start in range(0, rows, 128):
                stop = min(start + 128, rows)
                source = checkpoint_file.get_tensor(name) if vector else checkpoint_file.get_slice(name)[start:stop]
                w = source.float().numpy().astype(np.float64).reshape(-1, width // size, size)
                lo = w.min(2, keepdims=True)
                extent = w.max(2, keepdims=True) - lo
                probability = np.divide(w - lo, extent, out=np.full_like(w, .5), where=extent != 0).clip(1e-12, 1 - 1e-12)
                field = ((np.log(probability) - np.log1p(-probability)) / 2).clip(-4, 4)
                initial = np.rint((field + 4) * 4095 / 8).astype('<u2')
                theta = torch.from_numpy(initial.astype(np.float32) * np.float32(8 / 4095) - np.float32(4))
                raw = np.asarray(gradient if vector else gradient[start:stop]).reshape(w.shape)
                assert np.isfinite(raw).all()
                g = torch.from_numpy(raw.copy())
                # Exact CPU single-tensor first-step FP32 Adam primitives.
                m, v = torch.zeros_like(g), torch.zeros_like(g)
                m.lerp_(g, 1 - .9)
                v.mul_(.999).addcmul_(g, g, value=1 - .999)
                denominator = (v.sqrt() / math.sqrt(1 - .999)).add_(1e-8)
                theta.addcdiv_(m, denominator, value=-.003 / (1 - .9))
                theta.clamp_(-4, 4)
                new_theta = theta.numpy()
                assert np.isfinite(new_theta).all()
                replay = np.rint((new_theta.astype(np.float64) + 4) * 4095 / 8).clip(0, 4095).astype('<u2')
                np.testing.assert_array_equal(replay, (codes if vector else codes[start:stop]).reshape(w.shape))
                np.testing.assert_array_equal(lo.astype(np.float32), lower[start:stop])
                np.testing.assert_array_equal(extent.astype(np.float32), span[start:stop])
                for key, value in [('initial_codes', initial), ('codes', replay), ('lower', lo.astype('<f4')),
                                   ('span', extent.astype('<f4')), ('theta_after', new_theta.astype('<f4')), ('gradient', raw)]:
                    hashes[key].update(value.tobytes())
                parameter_changed += int(np.count_nonzero(replay != initial))
                parameter_nonzero += int(np.count_nonzero(raw))
                parameter_squared_norm += float(g.double().square().sum())
            for key, digest in hashes.items():
                target = gradrec if key == 'gradient' else rec
                assert digest.hexdigest() == target[key + '_sha256'], (name, key)
            assert parameter_nonzero == gradrec['gradient_nonzero']
            # Different reduction partitions can change the final FP64 rounding.
            assert math.isclose(parameter_squared_norm, gradrec['gradient_squared_sum_FP64'], rel_tol=1e-12, abs_tol=1e-24)
            assert gradrec['gradient_dtype'] == 'torch.float32' and gradrec['forward_uses'] >= 1
            assert rec['codes_changed'] == (parameter_changed > 0)
            total += param['numel']; groups += rec['groups']; changed += parameter_changed
            nonzero += parameter_nonzero; squared_norm += parameter_squared_norm
            changed_programs += int(parameter_changed > 0)
            (out / 'audit_progress.json').write_text(json.dumps({'last_parameter': name, 'coordinates_replayed': total}) + '\n')
    assert total == result['trainable_control_slots'] == 1728801536
    assert groups == 13531430 and result['optimizer_state_FP32_coordinates'] == 2 * total
    assert changed_programs == result['changed_programs'] == result['program_files'] == result['unique_encoders'] == 286
    aliases = result['module_aliases']
    assert set(aliases) == {name.removesuffix('.weight') for name in expected} | {'lm_head'}
    assert len(aliases) == result['weighted_modules'] == 287 and result['embedding_head_encoder_shared']
    for name, canonical in aliases.items():
        assert canonical == ('model.embed_tokens.weight' if name == 'lm_head' else name + '.weight')
    assert gradients['model.embed_tokens.weight']['forward_uses'] >= 2
    assert len({v['path'] for v in programs.values()}) == 286
    assert sha(out / 'arrays.npz') == result['arrays_sha256']
    arrays = np.load(out / 'arrays.npz')
    ids = np.load(root / 'adaptation_data/tokens.npz')['train_tokens'][:1, :16]
    np.testing.assert_array_equal(arrays['input_ids'], ids)
    losses = {}
    for stage in ['before', 'after']:
        y = arrays[stage + '_logits']
        assert y.shape == (1, 16, 128000) and y.dtype == np.float32 and np.isfinite(y).all()
        logits = torch.from_numpy(y[:, :-1].reshape(-1, 128000))
        targets = torch.from_numpy(ids[:, 1:].reshape(-1))
        f32 = F.cross_entropy(logits, targets).item()
        assert abs(f32 - result['diagnostic_train_loss_' + stage]) <= 4 * float(np.spacing(np.float32(f32)))
        losses[stage] = {'FP32': f32, 'stable_FP64': F.cross_entropy(logits.double(), targets).item()}
    assert result['saved_FP64_storage_bytes'] == sum(v['bytes'] for v in result['saved_storage_records'] if v['dtype'] == 'torch.float64') == 0
    assert result['saved_transient_storage_bytes'] == sum(v['bytes'] for v in result['saved_storage_records'] if not v['persistent_control_or_range'])
    audit = {'status': 'all_unique_coordinates_first_Adam_step_and_finite_program_exports_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'coordinates_replayed': total, 'range_groups': groups, 'changed_command_coordinates': changed,
             'changed_programs': changed_programs, 'nonzero_gradient_coordinates': nonzero,
             'raw_gradient_global_L2': math.sqrt(squared_norm), 'saved_loss_checks': losses,
             'scope': 'Independent first-step FP32 Adam primitive replay on every saved raw gradient from checkpoint initialization; updated theta hashes, all finite codes/fixed metadata, shared aliases, source/data/gradient/program hashes, saved logit loss arithmetic. Does not independently rerun the full model backward/forward, establish native trained quality, ACT12 training parity, heldout improvement or hardware advantage. One update on15train targets only.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
