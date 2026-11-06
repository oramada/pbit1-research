"""Independent metrics, model-unit bootstrap and scope/resource audit."""
import datetime
import json
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_high_budget.common import (
    ROOT, OUT, PARENT, BASE, WORD, PRECISION, METHODS, CONTRASTS,
    sha, write, guard, parent_closed)


def metric(z, y):
    maximum = z.max(1)
    loss = maximum+np.log(np.exp(z-maximum[:, None]).sum(1))-z[np.arange(len(y)), y]
    return np.array([np.count_nonzero(z.argmax(1) == y)/len(y), loss.sum()/len(y)])


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'completion.json').exists()
    s = json.loads((OUT/'statistics.json').read_text())
    assert s['protocol_sha256'] == sha(OUT/'protocol.json')
    for path, digest in s['artifact_sha256'].items():
        assert sha(path) == digest, path
    m = json.loads((OUT/'manifest.json').read_text())
    a = json.loads((OUT/'native_audit.json').read_text())
    assert m['complete'] and a['passed'] and a['manifest_sha256'] == sha(OUT/'manifest.json')
    assert len(m['rows']) == len(a['rows']) == 100
    assert a['public_updates'] == m['native_updates'] == 2011136000000
    assert a['counter_entries_exact'] == 6158250000
    recovered = {r['name']: r for r in a['rows']}
    assert len(recovered) == 100
    expected = {(method, model, 4096, seed) for method in range(2) for model in range(5) for seed in range(10)}
    assert {(r['method'], r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    y = np.load(BASE/'test_data.npz')['labels']
    values = np.empty((2, 5, 10, 2))
    saved_rows = {r['name']: r for r in s['rows']}
    assert len(saved_rows) == 100
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        proof = recovered[row['name']]
        assert proof['case_sha256'] == row['sha256']
        assert proof['all17500_array_receipts_exact'] and proof['all100000_final_logits_exact']
        with np.load(row['path']) as case:
            assert case['receipts'].dtype == np.uint8 and case['receipts'].shape == (1250, 14, 32)
            assert case['logits'].shape == (10000, 10) and case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
            value = metric(case['logits'], y)
        values[row['method'], row['model'], row['seed']] = value
        np.testing.assert_allclose(value, [saved_rows[row['name']]['accuracy'], saved_rows[row['name']]['mean_CE']], rtol=2e-12, atol=2e-13)
    arrays = np.load(OUT/'statistical_arrays.npz')
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-12, atol=2e-13)
    parent = np.load(PARENT/'statistical_arrays.npz')
    base = np.load(BASE/'statistical_arrays.npz')
    word = np.load(WORD/'statistical_arrays.npz')
    precision = np.load(PRECISION/'statistical_arrays.npz')
    controls = {}
    with np.load(PRECISION/'logits.npz') as z:
        for ci, name in [(1, 'BF16_weights_F64'), (3, 'BF16_Torch_CPU')]:
            controls[name] = np.array([metric(z[f'c{ci}_model{mi}'], y) for mi in range(5)])
            np.testing.assert_allclose(controls[name], precision['control_metrics'][ci], rtol=2e-12, atol=2e-13)
    # The means and FP32 controls are unchanged, already independently checked,
    # and bound by the complete parent receipts and current input seals.
    bm = base['bootstrap_models']
    bs = np.random.default_rng(p['bootstrap_seed']).integers(0, 10, (2, 5000, 5, 10))
    np.testing.assert_array_equal(bs, arrays['bootstrap_method_seeds'])
    seen = set()
    assert len(s['contrasts']) == 15
    for c in s['contrasts']:
        method, name = c['method_index'], c['reference']
        assert (method, name) not in seen and c['method'] == METHODS[method] and c['T'] == 4096
        seen.add((method, name))
        left = values[method]
        if name in controls:
            right, right_seeds = controls[name], None
        elif name == 'FP32_F64':
            right, right_seeds = base['baseline_metrics'][0], None
        elif name == 'own_mean':
            right, right_seeds = parent['mean_metrics'][method], None
        elif name == 'own_T1024':
            right, right_seeds = parent['values'][method, 2], parent['bootstrap_method_seeds'][method]
        elif name == 'original_T1024':
            right, right_seeds = base['values'][2], base['bootstrap_seeds']
        elif name == 'word_T256':
            right, right_seeds = word['values'][2], word['bootstrap_word_seeds']
        else:
            assert name == 'expected_T4096' and method == 1
            right, right_seeds = values[0], bs[0]
        dist = np.zeros((5000, 2))
        for unit in range(5):
            ids = bm[:, unit]
            dist += left[ids[:, None], bs[method, :, unit]].sum(1)/50
            dist -= right[ids]/5 if right_seeds is None else right[ids[:, None], right_seeds[:, unit]].sum(1)/50
        point = left.mean((0, 1))-(right.mean(0) if right.ndim == 2 else right.mean((0, 1)))
        np.testing.assert_allclose(point, c['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(dist, arrays[c['array_key']], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(dist, [.025, .975], axis=0).T,
                                   c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert seen == set(CONTRASTS)
    assert len(s['summaries']) == 2 and {r['method_index'] for r in s['summaries']} == {0, 1}
    for row in s['summaries']:
        method = row['method_index']
        np.testing.assert_allclose(values[method].mean((0, 1)), row['mean'], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(values[method].mean(1), row['model_means'], rtol=2e-12, atol=2e-13)
        np.testing.assert_array_equal(row['clipped_coordinates'], np.sum([r['clipped_coordinates'] for r in m['rows'] if r['method'] == method], axis=0))
    write(OUT/'statistics_audit.json', {'passed': True, 'case_metrics_checked': 100,
        'native_logits_checked': 10000000, 'BF16_paths_checked': 10, 'BF16_logits_checked': 1000000,
        'contrast_rows_checked': 15, 'bootstrap_scalar_values_checked': 150000,
        'statistics_sha256': sha(OUT/'statistics.json')})
    ledger = {'weight_pbits': 3928, 'activation_pbits': 0, 'deterministic_biases': 34,
        'nominal_parameter_payload_bytes': 4656, 'output_counter_bytes_per8imagebank': 197064,
        'retained_updates_per_native_or_public_pass': 2011136000000,
        'warmup_updates_per_native_or_public_pass': 491000000,
        'T4096_to_T1024_local_round_ratio': 4,
        'scope': 'Logical deployment payload/counters and scheduled evaluations, not total physical memory, flips, energy, area or latency. Prior training/validation/full cohorts/replays remain separate costs.'}
    write(OUT/'resource_ledger.json', ledger)
    report = ROOT.parent/'CIFAR_CNN_HIGH_BUDGET_REPORT.md'
    assert not report.exists()
    lines = ['# One-p-bit RGB inference at 4096 samples', '',
        'All100native cases and complete public-state replay passed; all100case metrics/10millionlogits,10BF16paths/1millionlogits and150000bootstrap scalar values independently checked. Five starting models, both original training methods, ten inference seeds, all10000historically consumed test images.', '',
        '| Method | Samples/layer | Accuracy (%) | CE |', '|---|---:|---:|---:|']
    for row in s['summaries']:
        lines.append(f"|{row['method']}|4096|{100*row['mean'][0]:.4f}|{row['mean'][1]:.8f}|")
    lines += ['', '| Method | Reference | Accuracy difference (pp) | Descriptive95%interval (pp) | CE difference |',
              '|---|---|---:|---|---:|']
    for c in s['contrasts']:
        lo, hi = c['descriptive_95_interval'][0]
        lines.append(f"|{c['method']}|{c['reference']}|{100*c['difference'][0]:+.4f}|[{100*lo:+.6f},{100*hi:+.6f}]|{c['difference'][1]:+.8f}|")
    lines += ['', p['statistics'], '', p['scope'], '',
        'One free stochastic weight node per normalized parameter remains separate from commands/ranges/bias/ACT8/counters/externalF64readout. Four times the local sample rounds ofT1024 are charged; no measured physical energy/latency/area advantage is inferred. Native and public replay each2011136000000retained+491000000warmup schedulednodeupdates. BF16CPUinternalaccumulator widths are not asserted. Comparable observed accuracy, a zero-crossing interval or passing implementation audit does not establish numerical BF16 precision or formal equivalence.', '',
        '[Protocol](cifar_cnn_native/probability_high_budget_evidence/protocol.json) · [All statistics](cifar_cnn_native/probability_high_budget_evidence/statistics.json) · [Public replay](cifar_cnn_native/probability_high_budget_evidence/native_audit.json) · [Independent audit](cifar_cnn_native/probability_high_budget_evidence/statistics_audit.json)', '',
        'The broader research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    guard(p)
    paths = [OUT/n for n in ['protocol.json', 'controls.json', 'mean_logits.npz', 'manifest.json',
        'native_audit.json', 'statistics.json', 'statistical_arrays.npz', 'statistics_audit.json', 'resource_ledger.json']]+[report]
    write(OUT/'completion.json', {'status': 'all100_T4096_RGB_cases_independently_audited_goal_active',
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All100T4096 RGB cases, BF16 contrasts and statistical evidence independently verified')


if __name__ == '__main__':
    main()
