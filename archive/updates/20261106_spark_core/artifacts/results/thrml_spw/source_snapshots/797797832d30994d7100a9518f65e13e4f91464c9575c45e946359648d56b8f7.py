"""Root review of the completed, unchanged RGB learned-probability cohort."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_cohort_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'review_completion.json').exists()
    completion = json.loads((OUT/'completion.json').read_text())
    terminal = 'full_probability_finetuned_RGB_independently_audited_goal_active'
    assert completion['status'] == terminal
    for path, digest in completion['artifact_sha256'].items():
        assert sha(path) == digest, path
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    assert pipeline['status'] == terminal and len(pipeline['stages']) == 6
    for row in pipeline['stages']:
        assert row['status'] == 'completed' and row['returncode'] == 0
        assert sha(OUT/'logs'/(row['stage']+'.log')) == row['log_sha256']
    m = json.loads((OUT/'manifest.json').read_text())
    a = json.loads((OUT/'native_audit.json').read_text())
    v = json.loads((OUT/'statistics_audit.json').read_text())
    # Producer seal stays immutable; separate audit binds that exact manifest.
    assert m['complete'] and m['independently_audited'] is False and a['passed'] and v['passed']
    assert a['manifest_sha256'] == sha(OUT/'manifest.json')
    assert len(a['rows']) == len(m['rows']) == 300
    assert a['counter_entries_exact'] == 18474750000
    assert a['public_updates'] == m['native_updates'] == 659904000000
    assert all(r['all17500_array_receipts_exact'] and r['all100000_final_logits_exact'] for r in a['rows'])
    assert v['case_metrics_checked'] == 300 and v['controls_checked'] == 10
    assert v['control_logits_checked'] == 1000000 and v['bootstrap_values_checked'] == 330000
    arrays = np.load(OUT/'statistical_arrays.npz')
    base = np.load(ROOT/'cohort_evidence/statistical_arrays.npz')
    word = np.load(ROOT/'word_cohort_evidence/statistical_arrays.npz')
    values, means = arrays['values'], arrays['mean_metrics']
    s = json.loads((OUT/'statistics.json').read_text())
    assert values.shape == (2, 3, 5, 10, 2)
    names = ['expected_minus_original', 'sampled_minus_original', 'sampled_minus_expected',
             'expected_minus_FP32', 'sampled_minus_FP32', 'expected_minus_own_mean',
             'sampled_minus_own_mean', 'expected_minus_INT4', 'sampled_minus_INT4',
             'expected_minus_word', 'sampled_minus_word']
    assert len(s['contrasts']) == 33
    seen = set()
    for c in s['contrasts']:
        bi, name = c['budget_index'], c['contrast']
        assert (bi, name) not in seen
        seen.add((bi, name))
        method = int(name.startswith('sampled'))
        suffix = name.split('_minus_')[1]
        reference = {'original': base['values'][bi].mean((0, 1)),
                     'expected': values[0, bi].mean((0, 1)),
                     'FP32': base['baseline_metrics'][0].mean(0),
                     'INT4': base['baseline_metrics'][4].mean(0),
                     'own_mean': means[method].mean(0),
                     'word': word['values'][bi].mean((0, 1))}[suffix]
        np.testing.assert_allclose(values[method, bi].mean((0, 1))-reference,
                                   c['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(arrays[f'{name}_budget{bi}'], [.025, .975], axis=0).T,
                                   c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
        assert c['T'] == [64, 256, 1024][bi]
    assert seen == {(bi, name) for bi in range(3) for name in names}
    report = ROOT.parent/'CIFAR_CNN_PROBABILITY_FINETUNE_REVIEW.md'
    assert not report.exists()
    lines = ['# Reviewed full RGB probability-aware training experiment', '',
             'All 300 native inference cases and independent public replay are complete. Completion-bound files, frozen source/input hashes, all six successful stages/log hashes, audit totals and all 33 contrast directions/units were checked in this root review. Independent graph/metrics/bootstrap verification already covers 1 million mean-control logits and 330000 bootstrap scalar values.', '',
             '| Method | Samples/layer | Accuracy (%) | Difference from FP32 (pp) | Descriptive 95% interval (pp) |',
             '|---|---:|---:|---:|---|']
    for row in s['summaries']:
        name = ('sampled' if row['method_index'] else 'expected')+'_minus_FP32'
        c = next(c for c in s['contrasts'] if c['contrast'] == name and c['T'] == row['T'])
        lo, hi = c['descriptive_95_interval'][0]
        lines.append(f"|{row['method']}|{row['T']}|{100*row['mean'][0]:.4f}|{100*c['difference'][0]:+.4f}|[{100*lo:+.6f}, {100*hi:+.6f}]|")
    lines += ['', 'FP32 accuracy is 56.2580%. At T=1024 the sample-aware method reaches 55.0146%, a loss of 1.2434 percentage points; expected-weight training reaches 55.0924%, a loss of 1.1656 points. These improve the original frozen 2.6936-point loss and fall inside the original approximate 1–2-point target descriptively. Neither establishes BF16 equivalence. At T=256 the sample-aware loss is still 3.3478 points. Every lower-budget result and both training methods remain reported.', '',
              'Five compact 16x16 RGB CNNs, ten inference seeds per condition, all 10000 historically consumed test images. Both methods receive eight extra epochs versus the original frozen/FP32/word references; expected-weight training controls that budget and parameterization. The sampled training estimator is a biased binomial-mean STE surrogate, not native THRML training. These unadjusted descriptive intervals are not fresh-data confirmation or equivalence tests.', '',
              'One free weight p-bit per normalized parameter, 3928 weight nodes; 34 biases, centers/radii, ACT8 spans, 8-bit bias commands and readout/accumulators remain separate costs. All 18474750000 integer counters and 30 million final logits were publicly regenerated and checked. Native production and public replay each use 659904000000 retained scheduled node updates plus 1473000000 warmup updates. These are neither physical state-change counts nor measured energy/latency/area.', '',
              '[Full results](CIFAR_CNN_PROBABILITY_FINETUNE_REPORT.md) · [All statistics](cifar_cnn_native/probability_cohort_evidence/statistics.json) · [Completion](cifar_cnn_native/probability_cohort_evidence/completion.json)', '',
              'The broader research goal and direct BF16 comparison remain open.']
    report.write_text('\n'.join(lines)+'\n')
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'contrast_rows_unit_checked': 33,
              'bootstrap_scalar_values_unit_checked': 330000,
              'artifact_sha256': {str(path): sha(path) for path in [OUT/'pipeline.json', OUT/'completion.json', report]},
              'scope': 'Root closure review, not independent extra inference replicas, BF16 equivalence or full goal completion.'}
    (OUT/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (OUT/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(OUT/'root_review.json'): sha(OUT/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('Full learned RGB cohort closure: all 300 cases, six stages and 33 contrast units reviewed')


if __name__ == '__main__':
    main()
