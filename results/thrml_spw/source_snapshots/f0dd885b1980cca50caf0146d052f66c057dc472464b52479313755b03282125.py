"""Independent RGB word metrics, mean controls, paired bootstrap and resources."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.word_cohort.common import ROOT, TRAIN, OUT, PARENT, sha, write, guard, parent_closed
from results.thrml_spw.fashion_cnn_native.cohort.common import CONTROLS


def main():
    import torch
    from results.thrml_spw.fashion_cnn_native.word_comparison.verify import mathematical_mean, torch_control, metric
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    s = json.loads((OUT/'statistics.json').read_text())
    required = {'manifest_sha256': OUT/'manifest.json', 'native_audit_sha256': OUT/'native_audit.json',
        'parent_statistics_sha256': PARENT/'statistics.json', 'parent_arrays_sha256': PARENT/'statistical_arrays.npz',
        'mean_logits_sha256': OUT/'mean_logits.npz', 'mean_weights_sha256': OUT/'mean_weights.npz',
        'statistical_arrays_sha256': OUT/'statistical_arrays.npz'}
    for key, path in required.items():
        assert sha(path) == s[key], path
    parent = np.load(PARENT/'statistical_arrays.npz')
    arrays = np.load(OUT/'statistical_arrays.npz')
    data = np.load(PARENT/'test_data.npz')
    x, y = data['inputs'], data['labels']
    saved, ww = np.load(OUT/'mean_logits.npz'), np.load(OUT/'mean_weights.npz')
    means = np.empty((5, 2))
    for mi, row in enumerate(json.loads((TRAIN/'manifest.json').read_text())['rows']):
        model = np.load(row['path'])
        weights = [ww[f'model{mi}_layer{li}'] for li in range(3)]
        for a, b in zip(weights, mathematical_mean(model)):
            np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-15)
        ref = np.concatenate([torch_control(x[first:first+128], model, weights) for first in range(0, 10000, 128)])
        np.testing.assert_allclose(ref, saved[f'model{mi}'], rtol=2e-12, atol=2e-11)
        np.testing.assert_array_equal(ref.argmax(1), saved[f'model{mi}'].argmax(1))
        means[mi] = metric(saved[f'model{mi}'], y)
    np.testing.assert_allclose(means, arrays['mean_metrics'], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(means.mean(0), s['mean_control_summary'], rtol=2e-12, atol=2e-13)
    m = json.loads((OUT/'manifest.json').read_text())
    a = json.loads((OUT/'native_audit.json').read_text())
    expected = {(mi, t, seed) for mi in range(5) for t in [16, 64, 256] for seed in range(10)}
    assert len(m['rows']) == len(a['rows']) == len(s['rows']) == 150
    assert {(r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    assert {(r['model'], r['T'], r['seed']) for r in a['rows']} == expected
    assert a['passed'] and a['complete'] and a['manifest_sha256'] == sha(OUT/'manifest.json')
    assert a['scheduled_updates'] == m['scheduled_updates'] == 329952000000
    assert a['counter_entries'] == m['counter_entries'] == 9237375000
    values = np.empty((3, 5, 10, 2))
    lookup = {r['name']: r for r in s['rows']}
    assert len(lookup) == 150
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            value = metric(case['logits'], y)
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
        bi = p['word_samples'].index(row['T'])
        values[bi, row['model'], row['seed']] = value
        np.testing.assert_allclose(value, [lookup[row['name']]['accuracy'], lookup[row['name']]['mean_CE']], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-12, atol=2e-13)
    bm, bs = parent['bootstrap_models'], parent['bootstrap_seeds']
    bw = np.random.default_rng(p['bootstrap_word_seed']).integers(0, 10, (5000, 5, 10))
    np.testing.assert_array_equal(bw, arrays['bootstrap_word_seeds'])
    assert len(s['contrasts']) == 12 and len(s['summaries']) == 3
    seen = set()
    for c in s['contrasts']:
        bi, name = c['budget_index'], c['contrast']
        assert (bi, name) not in seen
        seen.add((bi, name))
        word = sum(values[bi, bm[:, unit, None], bw[:, unit]].sum(1) for unit in range(5))/50
        if name == 'word_minus_single':
            reference = parent['values'][bi]
            ref = sum(reference[bm[:, unit, None], bs[:, unit]].sum(1) for unit in range(5))/50
            point = values[bi].mean((0, 1))-reference.mean((0, 1))
        else:
            assert name in p['contrasts']
            reference = means if name == 'word_minus_nominal_word_mean' else parent['baseline_metrics'][CONTROLS.index('FP32_F64' if name == 'word_minus_FP32' else 'ACT8_INT4_minmax')]
            ref = sum(reference[bm[:, unit]] for unit in range(5))/5
            point = values[bi].mean((0, 1))-reference.mean(0)
        delta = word-ref
        np.testing.assert_allclose(delta, arrays[f'{name}_budget{bi}'], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(point, c['difference'], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(np.quantile(delta, [.025, .975], axis=0).T, c['descriptive_95_interval'], rtol=2e-10, atol=2e-12)
    assert seen == {(bi, name) for bi in range(3) for name in p['contrasts']}
    for bi, row in enumerate(s['summaries']):
        assert row['T_word'] == p['word_samples'][bi] and row['T_single'] == 4*row['T_word']
        np.testing.assert_allclose(row['mean'], values[bi].mean((0, 1)), rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(row['model_means'], values[bi].mean(1), rtol=2e-12, atol=2e-13)
        assert row['clipped_coordinates'] == np.sum([r['clipped_coordinates'] for r in m['rows'] if r['T'] == row['T_word']], axis=0).tolist()
    guard(p)
    write(OUT/'statistics_audit.json', {'passed': True, 'case_metrics_checked': 150,
          'mean_control_logits_checked': 500000, 'weight_coordinates_checked': 19640,
          'bootstrap_scalar_values_checked': 120000, 'statistics_sha256': sha(OUT/'statistics.json')})
    ledger = {'single_weight_nodes': 3928, 'word_weight_nodes': 15712,
              'single_command_bytes': 3928, 'word_command_bytes': 1964,
              'shared_metadata_bytes': 728, 'single_parameter_payload_bytes': 4656,
              'word_parameter_payload_bytes': 2692, 'single_dynamic_state_bytes': 491,
              'word_dynamic_state_bytes': 1964, 'int32_counter_bytes_per8imagebank': 197064,
              'retained_updates_per_full_production_or_public_pass': 329952000000,
              'word_warmup_per_full_pass': 2946000000, 'single_warmup_per_full_pass': 736500000,
              'preflight_retained_updates_per_pass': 52792320, 'preflight_warmup_per_pass': 471360,
              'scope': 'Logical payload/state and software observer accounting; not physical area, energy, latency, controller/retention cost or optimized hardware.'}
    write(OUT/'resource_ledger.json', ledger)
    report = ROOT.parent/'CIFAR_CNN_MATCHED_WORD_REPORT.md'
    assert not report.exists()
    lines = ['# Full compact RGB matched-resource comparison', '',
        'All150word cases, full independent public replay and numerical/statistical checks passed. Same five compact3928-weight models and fixed historically consumed10000CIFAR images as the independently closed single-node cohort. Internal four-node positional reference; not an asserted Extropic hardware architecture. Moderate conventional baseline and16x16RGB/small-model scope remain explicit.', '',
        '| Single T | Word T | Word accuracy (%) | Word CE |', '|---:|---:|---:|---:|']
    for row in s['summaries']:
        lines.append(f"|{row['T_single']}|{row['T_word']}|{100*row['mean'][0]:.4f}|{row['mean'][1]:.8f}|")
    lines += ['', 'All12paired-model/independent-MC contrasts with descriptive95%intervals are retained. Images stay fixed; no fresh generalization, equivalence or favorable-budget selection claim. Nominal word means are nonlinear plug-ins, not E[stochastic CNN].', '',
        'Each full production/public pass uses329952000000retained updates; word warmup2946000000versus736500000single is additional. Four times the nodes and one quarter the rounds match retained software updates, not physical switching/energy/latency. Logical word/single payload2692/4656bytes and dynamic state1964/491bytes exclude controllers and other system costs. Both observers use197064bytes/bank int32counters. No systems win follows from these counts.', '',
        '[Protocol](cifar_cnn_native/word_cohort_evidence/protocol.json) · [All statistics](cifar_cnn_native/word_cohort_evidence/statistics.json) · [Public replay](cifar_cnn_native/word_cohort_evidence/native_audit.json) · [Independent statistics](cifar_cnn_native/word_cohort_evidence/statistics_audit.json). The broad research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/n for n in ['protocol.json', 'preflight.json', 'manifest.json', 'native_audit.json',
             'controls.json', 'mean_logits.npz', 'mean_weights.npz', 'statistics.json',
             'statistical_arrays.npz', 'statistics_audit.json', 'resource_ledger.json']]+[report]
    write(OUT/'completion.json', {'status': 'all150RGBword_cases_independently_audited_goal_active',
          'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All150RGBwordcases/120000bootstrap values independently checked', flush=True)


if __name__ == '__main__':
    main()
