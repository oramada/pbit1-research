"""Independent BF16 bit rounding, control graphs and same-fit contrast verification."""
import datetime
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from results.thrml_spw.cifar_cnn_native.same_fit_bf16.run import (
    ROOT, OUT, LOW, HIGH, BASE, NAMES, METHODS, PREFIXES, checkpoint,
    sha, write, guard, closed)
from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control


def round_bf16(raw):
    bits = np.asarray(raw, dtype=np.float32).view(np.uint32)
    rounded = (bits+np.uint32(0x7fff)+((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(float)


def independent(x, model, name):
    weights = [round_bf16(model[prefix+'_weight']) for prefix in PREFIXES]
    if name != 'BF16_Torch_CPU':
        return numpy_control(x, model, weights, name == 'BF16_weights_ACT8_F64')
    h = torch.as_tensor(round_bf16(x), dtype=torch.bfloat16)
    with torch.no_grad():
        for li, (prefix, weight) in enumerate(zip(PREFIXES, weights)):
            w = torch.as_tensor(weight, dtype=torch.bfloat16)
            b = torch.as_tensor(round_bf16(model[prefix+'_bias']), dtype=torch.bfloat16)
            h = F.max_pool2d(F.relu(F.conv2d(h, w, b, padding=1)), 2) if li < 2 else F.linear(h.flatten(1), w, b)
    assert h.dtype == torch.bfloat16
    return h.float().numpy().astype(float)


def compare(z, ref, name):
    if name == 'BF16_Torch_CPU':
        np.testing.assert_array_equal(z, ref)
    else:
        np.testing.assert_allclose(z, ref, rtol=2e-12, atol=2e-11)
    np.testing.assert_array_equal(z.argmax(1), ref.argmax(1))


def metric(z, y):
    top = z.max(1)
    losses = top+np.log(np.exp(z-top[:, None]).sum(1))-z[np.arange(len(y)), y]
    return np.array([np.count_nonzero(z.argmax(1) == y)/len(y), losses.sum()/len(y)])


def selfcheck():
    from results.thrml_spw.cifar_cnn_native.same_fit_bf16.run import output
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'selfcheck.json').exists()
    closed()
    x = np.load(ROOT/'training/validation.npz')['inputs'][:64]
    errors = {}
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            for prefix in PREFIXES:
                for suffix in ['weight', 'bias']:
                    raw = model[prefix+'_'+suffix]
                    actual = torch.as_tensor(raw, dtype=torch.float32).to(torch.bfloat16).double().numpy()
                    np.testing.assert_array_equal(round_bf16(raw), actual)
            for name in NAMES:
                z, ref = output(x, model, name), independent(x, model, name)
                compare(z, ref, name)
                errors[f'method{method}_model{mi}_{name}'] = float(np.max(abs(z-ref)))
    sources = sorted(Path(__file__).parent.glob('*.py'))
    write(OUT/'selfcheck.json', {'passed': True, 'validation_paths': 30,
        'inputs_per_path': 64, 'labels_used': False, 'max_errors': errors,
        'source_sha256': {str(path): sha(path) for path in sources},
        'scope': 'Validation-only graph/format checks; no task scoring, training or new native updates.'})
    print('All 30 validation-only BF16 paths pass independent checks')


def verify():
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    assert not (OUT/'completion.json').exists()
    s = json.loads((OUT/'statistics.json').read_text())
    e = json.loads((OUT/'evaluation.json').read_text())
    assert e['complete'] and e['control_paths'] == 30 and e['new_native_updates'] == 0
    assert e['protocol_sha256'] == s['protocol_sha256'] == sha(OUT/'protocol.json')
    assert e['logits_sha256'] == s['logits_sha256'] == sha(OUT/'logits.npz')
    assert e['statistics_sha256'] == sha(OUT/'statistics.json')
    assert s['statistical_arrays_sha256'] == sha(OUT/'statistical_arrays.npz')
    data = np.load(BASE/'test_data.npz'); x, y = data['inputs'], data['labels']
    arrays = np.load(OUT/'statistical_arrays.npz'); saved = np.load(OUT/'logits.npz')
    assert len(saved.files) == 30
    metrics = np.empty((2, 3, 5, 2)); errors = {}
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            for ci, name in enumerate(NAMES):
                key = f'method{method}_c{ci}_model{mi}'; z = saved[key]
                ref = np.concatenate([independent(x[first:first+128], model, name)
                                      for first in range(0, len(x), 128)])
                assert z.shape == (10000, 10) and np.isfinite(z).all()
                compare(z, ref, name); metrics[method, ci, mi] = metric(z, y)
                errors[key] = float(np.max(abs(z-ref)))
                print('Verified', key, flush=True)
    np.testing.assert_allclose(metrics, arrays['control_metrics'], rtol=2e-12, atol=2e-13)
    low = np.load(LOW/'statistical_arrays.npz'); high = np.load(HIGH/'statistical_arrays.npz')
    bm = np.load(BASE/'statistical_arrays.npz')['bootstrap_models']
    assert len(s['contrasts']) == 24
    seen = set()
    for c in s['contrasts']:
        method, ci, budget = c['method_index'], c['control_index'], c['T']
        bi = p['samples'].index(budget)
        assert (method, ci, budget) not in seen
        seen.add((method, ci, budget))
        assert c['control'] == NAMES[ci] and c['method'] == METHODS[method]
        values = low['values'][method, bi] if bi < 3 else high['values'][method]
        seeds = low['bootstrap_method_seeds'][method] if bi < 3 else high['bootstrap_method_seeds'][method]
        dist = np.zeros((5000, 2))
        for unit in range(5):
            model_ids = bm[:, unit]
            dist += values[model_ids[:, None], seeds[:, unit]].sum(1)/50
            dist -= metrics[method, ci, model_ids]/5
        point = values.mean((0, 1))-metrics[method, ci].mean(0)
        np.testing.assert_allclose(point, c['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(dist, arrays[c['array_key']], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(dist, [.025, .975], axis=0).T,
                                   c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert seen == {(m, ci, t) for m in range(2) for ci in range(3) for t in p['samples']}
    assert len(s['controls']) == 6
    assert {(r['method_index'], r['control_index']) for r in s['controls']} == {(m, ci) for m in range(2) for ci in range(3)}
    for row in s['controls']:
        np.testing.assert_allclose(metrics[row['method_index'], row['control_index']].mean(0), row['mean'], rtol=2e-12, atol=2e-13)
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'control_paths': 30,
        'logits_checked': 3000000, 'contrast_rows': 24, 'bootstrap_scalar_values_checked': 240000,
        'control_max_errors': errors, 'new_native_updates': 0, 'statistics_sha256': sha(OUT/'statistics.json')})
    report = ROOT.parent/'CIFAR_CNN_SAME_FIT_BF16_REPORT.md'; assert not report.exists()
    lines = ['# Same-fit BF16 comparison for probability-finetuned RGB CNNs', '',
        'Each BF16 control uses the exact learned mean weights from the corresponding one-p-bit fit. Both methods, all five models, all four sample budgets and all three BF16 controls are retained. Thirty full control graphs/3 million logits and all 24 contrasts/240000 bootstrap scalar values pass independent graph, format, metrics and statistics checks. Existing publicly verified stochastic outputs are reused; no new p-bit draws.', '',
        '| Training | Same-fit control | Mean accuracy (%) | Mean CE |', '|---|---|---:|---:|']
    for row in s['controls']:
        lines.append(f"|{row['method']}|{row['control']}|{100*row['mean'][0]:.4f}|{row['mean'][1]:.8f}|")
    lines += ['', '| Training | T | Same-fit control | P-bit minus control (pp) | Descriptive 95% interval (pp) |', '|---|---:|---|---:|---|']
    for c in s['contrasts']:
        lo, hi = c['descriptive_95_interval'][0]
        lines.append(f"|{c['method']}|{c['T']}|{c['control']}|{100*c['difference'][0]:+.4f}|[{100*lo:+.6f}, {100*hi:+.6f}]|")
    lines += ['', p['scope'], '',
        'The ACT8 weight-only BF16 control preserves the same activation discretization/spans and deterministic F64 readout/biases. The whole BF16 control changes activation arithmetic as declared. These controls separate training history within each comparison; they do not make a zero-crossing interval an equivalence test or establish numerical BF16 precision in a binary state. Sampling, metadata, commands and accumulation remain costs.', '',
        '[Protocol](cifar_cnn_native/same_fit_bf16_evidence/protocol.json) · [Statistics](cifar_cnn_native/same_fit_bf16_evidence/statistics.json) · [Independent audit](cifar_cnn_native/same_fit_bf16_evidence/evidence_audit.json)', '',
        'The broader research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/name for name in ['protocol.json', 'selfcheck.json', 'evaluation.json',
        'logits.npz', 'statistical_arrays.npz', 'statistics.json', 'evidence_audit.json']]+[report]
    write(OUT/'completion.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'same_fit_BF16_RGB_controls_independently_verified_goal_active',
        'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All 30 same-fit BF16 controls and 24 paired contrasts independently verified')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    {'selfcheck': selfcheck, 'verify': verify}[sys.argv[1]]()
