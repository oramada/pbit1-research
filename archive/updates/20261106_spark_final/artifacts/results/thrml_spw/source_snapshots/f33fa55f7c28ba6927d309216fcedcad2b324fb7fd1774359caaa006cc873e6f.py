"""Independent nominal means, neural controls, raw metrics and bootstrap verification."""
import datetime
import json
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.learned_word_cohort.common import (
    ROOT, OUT, LOW, HIGH, BASE, SAME, METHODS, CONTRASTS, checkpoint, sha, write,
    guard, closed, singles, independent_mean)


def metric(z, y):
    shifted = z-z.max(1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)), y]
    return np.array([np.count_nonzero(z.argmax(1) == y)/len(y), loss.sum()/len(y)])


def selfcheck():
    import torch
    from results.thrml_spw.cifar_cnn_native.learned_word_v2.adapter import mean_weights
    from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control
    from results.thrml_spw.fashion_cnn_native.word_comparison.verify import torch_control
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assert not (OUT/'controls_selfcheck.json').exists(); closed()
    x = np.load(ROOT/'training/validation.npz')['inputs'][:64]; rows = []
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method); canonical = mean_weights(model); other = independent_mean(model)
            for left, right in zip(canonical, other):
                np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-15)
            z, ref = numpy_control(x, model, canonical, True), torch_control(x, model, canonical)
            np.testing.assert_allclose(z, ref, rtol=2e-12, atol=2e-11)
            np.testing.assert_array_equal(z.argmax(1), ref.argmax(1))
            rows.append({'method': method, 'model': mi, 'weight_coordinates_checked': 3928,
                'logits_checked': 640, 'max_logit_error': float(np.max(abs(z-ref)))})
    write(OUT/'controls_selfcheck.json', {'passed': True, 'rows': rows, 'source_sha256': sha(__file__),
        'scope': 'Validation-only 10 nominal-word laws and distinct NumPy/Torch graphs; no labels/task scoring.'})


def run():
    import torch
    from results.thrml_spw.fashion_cnn_native.word_comparison.verify import torch_control
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    assert not (OUT/'completion.json').exists()
    s = json.loads((OUT/'statistics.json').read_text()); m = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    assert s['protocol_sha256'] == sha(OUT/'protocol.json')
    for path, digest in s['artifact_sha256'].items():
        assert sha(path) == digest
    assert m['complete'] and audit['passed'] and audit['complete']
    assert audit['manifest_sha256'] == sha(OUT/'manifest.json')
    expected = {(method, mi, t, seed) for method in range(2) for mi in range(5)
        for t in p['word_samples'] for seed in range(10)}
    assert len(m['rows']) == len(audit['rows']) == 400
    for collection in [m['rows'], audit['rows']]:
        assert {(r['method'], r['model'], r['T'], r['seed']) for r in collection} == expected
    assert audit['counter_entries'] == p['counter_entries']
    assert audit['scheduled_updates'] == m['scheduled_updates'] == p['retained_updates_per_pass']
    proofs = {r['name']: r for r in audit['rows']}; assert len(proofs) == 400
    data = np.load(BASE/'test_data.npz'); x, y = data['inputs'], data['labels']
    arrays = np.load(OUT/'statistical_arrays.npz'); saved_stats = {r['name']: r for r in s['rows']}
    assert len(saved_stats) == 400
    values = np.empty((2, 4, 5, 10, 2))
    for row in m['rows']:
        proof = proofs[row['name']]
        assert proof['sha256'] == row['sha256'] == sha(row['path']) and proof['all_receipts_and_final_logits_exact']
        with np.load(row['path']) as case:
            assert set(case.files) == {'receipts', 'logits', 'clipped_coordinates'}
            assert case['receipts'].dtype == np.uint8 and case['receipts'].shape == (1250, 14, 32)
            assert case['logits'].dtype == np.float64 and case['logits'].shape == (10000, 10)
            assert np.isfinite(case['logits']).all()
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
            value = metric(case['logits'], y)
        bi = p['word_samples'].index(row['T']); values[row['method'], bi, row['model'], row['seed']] = value
        np.testing.assert_allclose(value, [saved_stats[row['name']]['accuracy'], saved_stats[row['name']]['mean_CE']], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-12, atol=2e-13)
    means = np.empty((2, 5, 2)); errors = {}
    with np.load(OUT/'mean_logits.npz') as z, np.load(OUT/'mean_weights.npz') as ww:
        assert len(z.files) == 10 and len(ww.files) == 30
        for method in range(2):
            for mi in range(5):
                model = checkpoint(mi, method)
                canonical = [ww[f'method{method}_model{mi}_layer{li}'] for li in range(3)]
                for left, right in zip(canonical, independent_mean(model)):
                    np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-15)
                ref = np.concatenate([torch_control(x[first:first+128], model, canonical)
                    for first in range(0, len(x), 128)])
                key = f'method{method}_model{mi}'; actual = z[key]
                np.testing.assert_allclose(actual, ref, rtol=2e-12, atol=2e-11)
                np.testing.assert_array_equal(actual.argmax(1), ref.argmax(1))
                means[method, mi] = metric(actual, y); errors[key] = float(np.max(abs(actual-ref)))
    np.testing.assert_allclose(means, arrays['mean_metrics'], rtol=2e-12, atol=2e-13)
    single, single_seeds = singles(); bm = np.load(BASE/'statistical_arrays.npz')['bootstrap_models']
    bf16 = np.load(SAME/'statistical_arrays.npz')['control_metrics'][:, 1]
    seeds = np.random.default_rng(p['bootstrap_word_seed']).integers(0, 10, (2, 5000, 5, 10))
    np.testing.assert_array_equal(seeds, arrays['bootstrap_word_seeds'])
    assert len(s['contrasts']) == 24; seen = set()
    for c in s['contrasts']:
        method, bi, name = c['method_index'], c['budget_index'], c['contrast']
        assert (method, bi, name) not in seen; seen.add((method, bi, name))
        assert c['T_word'] == p['word_samples'][bi] and c['T_single'] == 4*c['T_word']
        assert c['method'] == METHODS[method]
        if name == 'single_minus_word':
            left, right = single[method, bi], values[method, bi]
            ls, rs = single_seeds[bi][method], seeds[method]
        else:
            assert name in CONTRASTS
            left, ls, rs = values[method, bi], seeds[method], None
            right = means[method] if name == 'word_minus_nominal_word_mean' else bf16[method]
        distribution = np.zeros((5000, 2))
        for unit in range(5):
            ids = bm[:, unit]
            distribution += left[ids[:, None], ls[:, unit]].sum(1)/50
            distribution -= right[ids]/5 if rs is None else right[ids[:, None], rs[:, unit]].sum(1)/50
        point = left.mean((0, 1))-(right.mean(0) if rs is None else right.mean((0, 1)))
        np.testing.assert_allclose(point, c['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(distribution, arrays[c['array_key']], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(distribution, [.025, .975], axis=0).T,
            c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert seen == {(method, bi, name) for method in range(2) for bi in range(4) for name in CONTRASTS}
    assert len(s['summaries']) == 8
    assert {(r['method_index'], r['T_word']) for r in s['summaries']} == {(m, t) for m in range(2) for t in p['word_samples']}
    for row in s['summaries']:
        method = row['method_index']; bi = p['word_samples'].index(row['T_word'])
        np.testing.assert_allclose(values[method, bi].mean((0, 1)), row['mean'], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(values[method, bi].mean(1), row['model_means'], rtol=2e-12, atol=2e-13)
        np.testing.assert_array_equal(row['clipped_coordinates'], np.sum([r['clipped_coordinates'] for r in m['rows']
            if r['method'] == method and r['T'] == row['T_word']], axis=0))
    np.testing.assert_allclose(means.mean(1), s['mean_control_summary'], rtol=2e-12, atol=2e-13)
    guard(p)
    write(OUT/'statistics_audit.json', {'passed': True, 'case_metrics_checked': 400,
        'native_logits_checked': 40000000, 'control_logits_checked': 1000000,
        'mean_weight_coordinates_checked': 39280, 'bootstrap_scalar_values_checked': 240000,
        'contrasts_checked': 24, 'control_max_errors': errors, 'statistics_sha256': sha(OUT/'statistics.json')})
    ledger = {'single_weight_pbits': 3928, 'word_weight_pbits': 15712, 'deterministic_biases': 34,
        'retained_evaluations_per_representation': p['retained_updates_per_pass'],
        'word_warmup_per_pass': p['warmup_updates_per_pass'], 'single_inherited_warmup_total': 1964000000,
        'word_counter_bytes_per_bank': 197064, 'word_nominal_parameter_payload_bytes': 2692,
        'single_nominal_parameter_payload_bytes': 4656,
        'scope': 'Logical payload/counters and scheduled evaluations only; dynamic states, programming precision, buffers, area, energy, readout, accumulation and retained control hardware are separate. No measured physical speedup or total-memory advantage.'}
    write(OUT/'resource_ledger.json', ledger)
    report = ROOT.parent/'CIFAR_CNN_LEARNED_WORD_MATCHED_REPORT.md'; assert not report.exists()
    lines = ['# Same-fit single-p-bit versus four-p-bit RGB inference', '',
        'All400four-node native cases and independent public-state replay passed. All40millionstoredlogits,1millionnominal-control logits,39280mean-weightcoordinates and240000bootstrap scalar values independently checked. Both methods/all five starting models/all ten MC seeds/all four budgets retained. The one-node results are the unchanged previously audited inference of the exact same learned fits, centers/radii/biases and ACT8 spans.', '',
        '| Training | Single T | Word T | Single accuracy (%) | Word accuracy (%) | Single minus word (pp) | Descriptive95%interval (pp) |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in s['summaries']:
        method = row['method_index']; bi = p['word_samples'].index(row['T_word'])
        c = next(c for c in s['contrasts'] if c['method_index'] == method and c['budget_index'] == bi and c['contrast'] == 'single_minus_word')
        lo, hi = c['descriptive_95_interval'][0]
        lines.append(f"|{row['method']}|{row['T_single']}|{row['T_word']}|{100*single[method, bi, :, :, 0].mean():.4f}|{100*row['mean'][0]:.4f}|{100*c['difference'][0]:+.4f}|[{100*lo:+.6f}, {100*hi:+.6f}]|")
    lines += ['', p['statistics'], '', p['scope'], '',
        'Retained node evaluations are matched; word warmup is four times the corresponding inherited single-node warmup and is reported separately. The internal positional reference has four free finite-field nodes per normalized weight, not an asserted Extropic hardware weight format. Nominal word-mean nonlinear controls are not E[finite-sample CNN]. Same-fit BF16 comparisons are included among all24contrasts. No formal equivalence, fresh-data confirmation, per-weight BF16 numerical precision or measured physical improvement follows.', '',
        '[Protocol](cifar_cnn_native/learned_word_cohort_evidence/protocol.json) · [All statistics](cifar_cnn_native/learned_word_cohort_evidence/statistics.json) · [Independent audit](cifar_cnn_native/learned_word_cohort_evidence/statistics_audit.json)', '',
        'The broader research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/name for name in ['protocol.json', 'controls.json', 'mean_logits.npz', 'mean_weights.npz',
        'manifest.json', 'native_audit.json', 'statistics.json', 'statistical_arrays.npz',
        'statistics_audit.json', 'resource_ledger.json']]+[report]
    write(OUT/'completion.json', {'status': 'all400_samefit_word_cases_independently_verified_goal_active',
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All400same-fit matched-work cases and24contrasts independently verified')


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'run': run}[sys.argv[1]]()
