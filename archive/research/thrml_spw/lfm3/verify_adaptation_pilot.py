"""Independent checkpoint/gradient Adam replay and validation-loss audit.

Full raw-gradient replay checks optimization arithmetic and serialized state.
It does not recompute full-model derivatives or native sampled inference.
"""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['mean', 'binomial', 'baseline'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_pilot' / args.arm
    protocol = json.loads((root / 'adaptation_pilot_protocol.json').read_text())
    arm_protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    assert arm_protocol['arm'] == result['arm'] == args.arm
    assert arm_protocol['shared_protocol_sha256'] == sha(root / 'adaptation_pilot_protocol.json')
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for path, digest in protocol['prerequisites_sha256'].items():
        assert sha(root / path) == digest
    assert result['status'] == 'all_parameter_probability_adaptation_and_validation_surrogate_complete'
    steps = 0 if args.arm == 'baseline' else protocol['steps']
    assert result['training_steps'] == steps and result['training_targets'] == steps * 64
    assert sha(out / 'program_package.json') == result['program_package_sha256']
    package = json.loads((out / 'program_package.json').read_text())
    assert package['training_steps'] == steps and len(package['programs']) == 286
    assert sha(out / 'module_aliases.json') == package['module_aliases_sha256']
    assert sha(out / 'initial_fingerprints.json') == package['initial_fingerprints_sha256']
    assert sha(out / 'training_rng_state.npy') == package['training_rng_state_sha256']
    aliases = json.loads((out / 'module_aliases.json').read_text())
    initial = json.loads((out / 'initial_fingerprints.json').read_text())
    meta = {v['name']: v for v in json.loads((root / 'meta_inventory.json').read_text())['parameters']}
    programs = {v['name']: v for v in package['programs']}
    assert len(programs) == len(initial) == len(meta) == 286 and set(programs) == set(initial) == set(meta)
    assert set(aliases) == {v.removesuffix('.weight') for v in meta} | {'lm_head'}
    assert len(aliases) == result['weighted_modules'] == 287
    for name, canonical in aliases.items():
        assert canonical == ('model.embed_tokens.weight' if name == 'lm_head' else name + '.weight')
    updates = [json.loads((out / f'step{step:03d}.json').read_text()) for step in range(1, steps + 1)]
    update_gradients = []
    for step, rec in enumerate(updates, 1):
        assert rec['step'] == step and rec['train_window_index'] == step - 1 and rec['targets'] == 64
        assert math.isfinite(rec['loss_FP32']) and math.isfinite(rec['gradient_L2'])
        records = {v['name']: v for v in rec['gradients']}
        assert len(records) == 286 and set(records) == set(meta)
        update_gradients.append(records)
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    assert sha(checkpoint) == json.loads((root / 'preparation_inventory.json').read_text())['checkpoint_files']['model.safetensors']
    norms = np.zeros(steps, dtype=np.float64)
    total = groups = changed = 0
    with safe_open(str(checkpoint), framework='pt') as source:
        for name, param in meta.items():
            rec = programs[name]
            assert rec['shape'] == param['shape'] and rec['numel'] == param['numel']
            program_path, state_path = Path(rec['program']), Path(rec['checkpoint'])
            assert sha(program_path) == rec['program_sha256']
            assert sha(state_path) == rec['checkpoint_sha256']
            with np.load(program_path) as f:
                assert set(f.files) == {'codes', 'lower', 'span', 'shape', 'group_size', 'format_version'}
                assert f['shape'].tolist() == param['shape'] and int(f['format_version']) == 1
                codes, lower, span = [f[k] for k in ['codes', 'lower', 'span']]
                size_saved = int(f['group_size'])
            with np.load(state_path) as f:
                state = {k: f[k] for k in f.files}
            assert set(state) == ({'theta', 'exp_avg', 'exp_avg_sq', 'step'} if steps else {'theta'})
            for key, value in state.items():
                assert value.dtype == np.float32 and np.isfinite(value).all()
                assert hashlib.sha256(value.tobytes()).hexdigest() == rec['state_array_hashes'][key]
                assert value.shape == (() if key == 'step' else tuple(param['shape']))
            if steps:
                assert float(state['step']) == steps
            assert codes.dtype == np.uint16 and np.max(codes) <= 4095 and codes.shape == state['theta'].shape
            assert lower.dtype == span.dtype == np.float32
            vector = len(param['shape']) == 1
            rows = 1 if vector else param['shape'][0]
            width = param['shape'][0] if vector else math.prod(param['shape'][1:])
            size = min(128, width)
            assert size == size_saved and width % size == 0
            assert lower.shape == span.shape == (rows, width // size, 1)
            raw = []
            for gradient_records in update_gradients:
                g = gradient_records[name]
                path = Path(g['path'])
                assert sha(path) == g['sha256'] and g['numel'] == param['numel']
                array = np.load(path, mmap_mode='r')
                assert array.dtype == np.float32 and list(array.shape) == param['shape']
                raw.append(array)
            grad_hashes = [hashlib.sha256() for _ in raw]
            nonzero = np.zeros(steps, dtype=np.int64)
            hashes = {key: hashlib.sha256() for key in ['initial_codes', 'codes', 'lower', 'span', 'theta']}
            for start in range(0, rows, 128):
                stop = min(start + 128, rows)
                tensor = source.get_tensor(name) if vector else source.get_slice(name)[start:stop]
                w = tensor.float().numpy().astype(np.float64).reshape(-1, width // size, size)
                lo = w.min(2, keepdims=True); extent = w.max(2, keepdims=True) - lo
                p = np.divide(w - lo, extent, out=np.full_like(w, .5), where=extent != 0).clip(1e-12, 1 - 1e-12)
                h = ((np.log(p) - np.log1p(-p)) / 2).clip(-4, 4)
                initial_codes = np.rint((h + 4) * 4095 / 8).astype('<u2')
                theta = torch.from_numpy(initial_codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4))
                m, v = torch.zeros_like(theta), torch.zeros_like(theta)
                for index, array in enumerate(raw):
                    values = np.asarray(array if vector else array[start:stop]).reshape(w.shape)
                    assert np.isfinite(values).all()
                    grad_hashes[index].update(values.tobytes())
                    nonzero[index] += np.count_nonzero(values)
                    g = torch.from_numpy(values.copy())
                    norms[index] += float(g.double().square().sum())
                    m.lerp_(g, 1 - .9)
                    v.mul_(.999).addcmul_(g, g, value=1 - .999)
                    step = index + 1
                    denominator = (v.sqrt() / math.sqrt(1 - .999 ** step)).add_(1e-8)
                    theta.addcdiv_(m, denominator, value=-protocol['learning_rate'] / (1 - .9 ** step))
                    theta.clamp_(-4, 4)
                decoded = theta.numpy()
                replay_codes = np.rint((decoded.astype(np.float64) + 4) * 4095 / 8).clip(0, 4095).astype('<u2')
                for key, actual in [('theta', decoded), ('exp_avg', m.numpy()), ('exp_avg_sq', v.numpy())]:
                    if key == 'theta' or steps:
                        np.testing.assert_array_equal(actual, (state[key] if vector else state[key][start:stop]).reshape(w.shape))
                np.testing.assert_array_equal(replay_codes, (codes if vector else codes[start:stop]).reshape(w.shape))
                np.testing.assert_array_equal(lo.astype(np.float32), lower[start:stop])
                np.testing.assert_array_equal(extent.astype(np.float32), span[start:stop])
                for key, value in [('initial_codes', initial_codes), ('codes', replay_codes), ('lower', lo.astype('<f4')),
                                   ('span', extent.astype('<f4')), ('theta', decoded)]:
                    hashes[key].update(value.tobytes())
                changed += int(np.count_nonzero(replay_codes != initial_codes))
            for index, digest in enumerate(grad_hashes):
                assert digest.hexdigest() == update_gradients[index][name]['array_sha256']
                assert int(nonzero[index]) == update_gradients[index][name]['nonzero']
            for key in ['initial_codes', 'codes', 'lower', 'span']:
                assert hashes[key].hexdigest() == rec[key + '_sha256']
            for key in ['codes', 'lower', 'span']:
                assert hashes['initial_codes' if key == 'codes' else key].hexdigest() == initial[name][key + '_sha256']
            assert hashes['theta'].hexdigest() == rec['state_array_hashes']['theta']
            total += param['numel']; groups += lower.size
            (out / 'audit_progress.json').write_text(json.dumps({'coordinates_replayed': total, 'last_parameter': name}) + '\n')
    assert total == result['unique_controls'] == 1728801536 and groups == 13531430
    assert result['unique_encoders'] == 286
    for value, rec in zip(norms, updates, strict=True):
        assert math.isclose(math.sqrt(value), rec['gradient_L2'], rel_tol=1e-12)
    validation = np.load(root / 'adaptation_data/tokens.npz')['validation_tokens']
    summaries = [result['before_mean'], result['after_mean'], *result['after_binomial']]
    assert len(result['after_binomial']) == len(protocol['validation_seeds']) == 5
    assert [v['seed'] for v in result['after_binomial']] == protocol['validation_seeds']
    if args.arm == 'baseline':
        assert result['BF16_validation'] is not None
        summaries.append(result['BF16_validation'])
    else:
        assert result['BF16_validation'] is None
    validation_checks = []
    for summary in summaries:
        losses = []
        assert len(summary['batches']) == 4
        for batch, rec in enumerate(summary['batches']):
            path = Path(rec['path'])
            assert sha(path) == rec['sha256']
            with np.load(path) as a:
                np.testing.assert_array_equal(a['input_ids'], validation[batch * 8:(batch + 1) * 8])
                logits, ids = a['logits'], a['input_ids']
                assert logits.dtype == np.float32 and logits.shape == (8, 65, 128000)
                assert np.isfinite(logits).all()
                z = logits[:, :-1].astype(np.float64)
                mx = z.max(-1)
                nll = mx + np.log(np.exp(z - mx[..., None]).sum(-1)) - np.take_along_axis(z, ids[:, 1:, None], axis=-1)[..., 0]
                np.testing.assert_allclose(a['token_NLL'], nll, rtol=2e-12, atol=2e-12)
                losses.append(nll)
        mean = float(np.concatenate(losses).mean())
        assert summary['targets'] == 2048
        assert math.isclose(mean, summary['mean_NLL_FP64'], rel_tol=2e-12, abs_tol=2e-12)
        assert math.isclose(math.exp(mean), summary['subset_perplexity'], rel_tol=2e-12)
        validation_checks.append({'label': summary.get('label', 'BF16'), 'mean_NLL_FP64': mean})
    if args.arm == 'baseline':
        assert changed == 0
        for before, after in zip(result['before_mean']['batches'], result['after_mean']['batches'], strict=True):
            with np.load(before['path']) as b, np.load(after['path']) as a:
                np.testing.assert_array_equal(b['logits'], a['logits'])
    audit = {'status': 'all_coordinates_Adam_trajectory_programs_and_validation_losses_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'), 'arm': args.arm,
             'unique_controls': total, 'steps': steps, 'control_updates_replayed': total * steps,
             'changed_commands': changed, 'validation_checks': validation_checks,
             'scope': 'Independent raw-gradient multi-step FP32 Adam replay, every final theta/moment/code/range, shared aliases and saved validation logit losses. Does not rerun full derivatives, sampled training draws, or model forwards; independent component STE algebra audited upstream. Validation is nominal-IID Torch surrogate, not officialTHRML quality, generalization or BF16 equivalence.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
