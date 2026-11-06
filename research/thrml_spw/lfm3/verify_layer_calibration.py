"""Independent gradient-based variance, geometry and allocation-budget audit."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def near(actual, expected):
    assert abs(actual - expected) <= 1e-12 * max(abs(expected), 1e-30), (actual, expected)


def main():
    torch.set_num_threads(4)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'layer_calibration'
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    assert r['status'] == 'full_model_separate_use_gradient_calibration_and_budget_candidate_complete'
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['prerequisites_sha256'].items():
        assert sha(root / name) == digest
    records = {v['name']: v for v in json.loads((root / 'adaptation_pilot/mean/program_package.json').read_text())['programs']}
    meta = {v['name']: v for v in json.loads((root / 'meta_inventory.json').read_text())['parameters']}
    aliases = r['module_aliases']
    assert set(records) == set(meta) and len(records) == 286
    assert set(aliases) == {v.removesuffix('.weight') for v in meta} | {'lm_head'} and len(aliases) == 287
    for module, canonical in aliases.items():
        assert canonical == ('model.embed_tokens.weight' if module == 'lm_head' else module + '.weight')
    for rec in records.values():
        assert sha(Path(rec['program'])) == rec['program_sha256']
    inputs = np.load(root / 'adaptation_data/tokens.npz')['train_tokens'][16:32, :65]
    score_batches, cost_batches = [], []
    total_gradient_values = 0
    for batch in range(2):
        path = out / f'batch{batch}.json'
        assert sha(path) == r['batch_records_sha256'][str(path)]
        b = json.loads(path.read_text())
        assert b['batch'] == batch
        assert sha(Path(b['outputs'])) == b['outputs_sha256']
        with np.load(b['outputs']) as a:
            ids, logits = a['input_ids'], a['logits']
            np.testing.assert_array_equal(ids, inputs[batch * 8:(batch + 1) * 8])
            assert logits.dtype == np.float32 and logits.shape == (8, 65, 128000) and np.isfinite(logits).all()
            f32 = F.cross_entropy(torch.from_numpy(logits[:, :-1].reshape(-1, 128000)), torch.from_numpy(ids[:, 1:].reshape(-1))).item()
            assert abs(f32 - b['loss_FP32_backward']) <= 4 * float(np.spacing(np.float32(f32)))
            z = logits[:, :-1].astype(np.float64)
            mx = z.max(-1)
            nll = mx + np.log(np.exp(z - mx[..., None]).sum(-1)) - np.take_along_axis(z, ids[:, 1:, None], -1)[..., 0]
            np.testing.assert_allclose(a['token_NLL'], nll, rtol=1e-12, atol=1e-12)
            near(b['mean_NLL_FP64'], float(nll.mean()))
        captured = {v['module']: v for v in b['gradient_records']}
        assert len(b['gradient_records']) == len(captured) == 287 and set(captured) == set(aliases)
        scores, costs = {}, {}
        for module, rec in captured.items():
            canonical = aliases[module]
            assert rec['canonical'] == canonical
            shape = meta[canonical]['shape']
            kind = ('embedding' if module == 'model.embed_tokens' else 'conv' if len(shape) == 3
                    else 'linear' if len(shape) == 2 else 'elementwise')
            assert rec['kind'] == r['kinds'][module] == kind
            gpath = Path(rec['gradient_path'])
            assert sha(gpath) == rec['file_sha256']
            gradient = np.load(gpath, mmap_mode='r')
            assert gradient.dtype == np.float64 and gradient.flags.c_contiguous
            with np.load(records[canonical]['program']) as program:
                codes, span = program['codes'], program['span']
            if kind == 'embedding':
                rows = np.unique(ids)
                assert rec['rows'] == rows.tolist()
                codes, span = codes[rows], span[rows]
                fanin, channels = 1, len(rows) * shape[1]
            else:
                assert rec['rows'] is None
                fanin, channels = (shape[1], shape[0]) if kind == 'linear' else (7, shape[0]) if kind == 'conv' else (1, shape[0])
            assert list(gradient.shape) == rec['shape'] == list(codes.shape)
            assert rec['numel'] == gradient.size
            vector = gradient.ndim == 1
            count = 1 if vector else gradient.shape[0]
            width = gradient.size if vector else math.prod(gradient.shape[1:])
            size = min(128, width)
            coefficient = norm = 0.
            h = hashlib.sha256()
            for start in range(0, count, 128):
                end = min(start + 128, count)
                values = gradient if vector else gradient[start:end]
                assert np.isfinite(values).all()
                h.update(values.tobytes())
                cc = codes if vector else codes[start:end]
                field = cc.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
                probability = np.exp(-np.logaddexp(0, -2 * field.astype(np.float64))).reshape(end - start, width // size, size)
                squared = values.reshape(probability.shape) ** 2
                coefficient += float(np.sum(squared * span[start:end].astype(np.float64) ** 2 * probability * (1 - probability)))
                norm += float(squared.sum())
            assert h.hexdigest() == rec['array_sha256']
            near(rec['variance_coefficient'], coefficient); near(rec['squared_gradient_norm'], norm)
            scores[module] = coefficient
            costs[module] = fanin * (((channels + 63) // 64) * 64)
            assert b['costs_including_padding'][module] == costs[module]
            total_gradient_values += gradient.size
        score_batches.append(scores); cost_batches.append(costs)
        (out / 'audit_progress.json').write_text(json.dumps({'batches_verified': batch + 1, 'weight_gradient_values_verified': total_gradient_values}) + '\n')
    scores = {name: sum(b[name] for b in score_batches) / 2 for name in aliases}
    costs = {name: (cost_batches[0][name] + cost_batches[1][name]) // 2 for name in aliases}
    for name in aliases:
        near(r['scores'][name], scores[name]); assert r['mean_costs'][name] == costs[name]
        if name != 'model.embed_tokens':
            assert cost_batches[0][name] == cost_batches[1][name]
    # Replay the published FP64 policy inputs after independently checking
    # their algebra; a few rounding bits must not redefine policy tie inputs.
    scores = {name: float(r['scores'][name]) for name in aliases}
    # Independent greedy replay using explicitly calculated objective changes.
    counts = {name: 1024 if name == 'model.embed_tokens' else 16 for name in aliases}
    cap = 1024 * sum(costs.values())
    used = sum(costs[name] * counts[name] for name in aliases)
    while True:
        options = []
        for name in sorted(aliases):
            old = counts[name]
            if name == 'model.embed_tokens' or old == 4096 or scores[name] <= 0:
                continue
            cost = costs[name] * old
            if used + cost <= cap:
                gain = scores[name] / old - scores[name] / (2 * old)
                options.append((-gain / cost, name, cost))
        if not options:
            break
        _, name, cost = min(options)
        counts[name] *= 2; used += cost
    proposal = r['proposal']
    assert counts == proposal['greedy_samples']
    assert used == proposal['greedy_cost'] <= cap == proposal['cost_cap']
    assert proposal['unused_cap'] == cap - used
    near(proposal['unused_fraction'], (cap - used) / cap)
    uniform = sum(scores[name] / 1024 for name in aliases)
    greedy = sum(scores[name] / counts[name] for name in aliases)
    near(proposal['uniform_proxy'], uniform); near(proposal['greedy_proxy'], greedy)
    selected = counts if greedy <= uniform else {name: 1024 for name in aliases}
    assert proposal['selected_samples'] == selected
    assert proposal['selected_cost'] == sum(costs[name] * selected[name] for name in aliases) <= cap
    assert proposal['selected_policy'] == ('greedy' if greedy <= uniform else 'uniform')
    near(proposal['selected_proxy'], min(greedy, uniform))
    assert r['unique_parameter_coordinates'] == 1728801536 and r['calibration_targets'] == 1024
    assert r['module_uses_per_batch'] == 287
    audit = {'status': 'all_module_use_variance_coefficients_raw_gradients_and_budget_candidate_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'weight_gradient_values_verified': total_gradient_values, 'module_uses_per_batch': 287,
        'selected_policy': proposal['selected_policy'], 'proxy_ratio_to_uniform': min(greedy, uniform) / uniform if uniform else None,
        'unused_cap_fraction': proposal['unused_fraction'],
        'scope': 'Independent saved-gradient/finite-program variance algebra, tied uses separated, rawfile/content hashes, saved calibrationlosses, nodepaddinggeometry and greedybudget replay. Does not independently rerun fullmodel derivatives or establish true nonlinear lossvariance, expected taskquality, native allocation gains or BF16 equivalence.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
