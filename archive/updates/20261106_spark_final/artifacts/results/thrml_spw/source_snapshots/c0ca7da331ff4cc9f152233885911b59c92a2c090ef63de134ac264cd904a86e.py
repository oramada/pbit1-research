"""Root closure and complete unit-checked RGB spatial/temporal comparison."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'word_cohort_evidence'
BASE = ROOT/'cohort_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'review_completion.json').exists()
    for folder in [BASE, OUT]:
        closed = json.loads((folder/'completion.json').read_text())
        for path, expected in closed['artifact_sha256'].items():
            assert sha(path) == expected, path
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, expected in p[group].items():
            assert sha(path) == expected, path
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    assert pipeline['status'] == 'all150RGBword_cases_independently_audited_goal_active'
    assert all(r['status'] == 'completed' and r['returncode'] == 0 for r in pipeline['stages'])
    for row in pipeline['stages']:
        assert row['log_sha256'] == sha(OUT/'logs'/(row['stage']+'.log'))
    a = json.loads((OUT/'native_audit.json').read_text())
    m = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'statistics_audit.json').read_text())
    assert a['passed'] and a['complete'] and m['complete']
    assert a['manifest_sha256'] == sha(OUT/'manifest.json')
    expected = {(mi, t, seed) for mi in range(5) for t in [16, 64, 256] for seed in range(10)}
    assert len(m['rows']) == len(a['rows']) == 150
    assert {(r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    assert all(r['all_receipts_and_final_logits_exact'] for r in a['rows'])
    assert a['counter_entries'] == 9237375000 and a['scheduled_updates'] == 329952000000
    assert audit['passed'] and audit['case_metrics_checked'] == 150 and audit['bootstrap_scalar_values_checked'] == 120000
    s = json.loads((OUT/'statistics.json').read_text())
    arrays, base = np.load(OUT/'statistical_arrays.npz'), np.load(BASE/'statistical_arrays.npz')
    assert arrays['values'].shape == base['values'].shape == (3, 5, 10, 2)
    contrasts, seen = [], set()
    for c in s['contrasts']:
        bi, name = c['budget_index'], c['contrast']
        assert (bi, name) not in seen
        seen.add((bi, name))
        if name == 'word_minus_single':
            reference = base['values'][bi].mean((0, 1))
        elif name == 'word_minus_FP32':
            reference = base['baseline_metrics'][0].mean(0)
        elif name == 'word_minus_INT4_minmax':
            reference = base['baseline_metrics'][4].mean(0)
        else:
            assert name == 'word_minus_nominal_word_mean'
            reference = arrays['mean_metrics'].mean(0)
        point = arrays['values'][bi].mean((0, 1))-reference
        interval = np.quantile(arrays[f'{name}_budget{bi}'], [.025, .975], axis=0).T
        np.testing.assert_allclose(point, c['difference'], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(interval, c['descriptive_95_interval'], rtol=2e-10, atol=2e-12)
        contrasts.append((bi, name, point, interval))
    assert seen == {(bi, name) for bi in range(3) for name in p['contrasts']}
    assert len(contrasts) == 12
    report = ROOT.parent/'CIFAR_CNN_MATCHED_WORD_REVIEW.md'
    assert not report.exists()
    lines = ['# Reviewed RGB one-p-bit versus four-p-bit comparison', '',
        'All150word cases, full public replay, mean controls and independent statistics passed. Same five frozen compactRGBmodels, all10000fixed historically consumed test images and ten inference seeds perbudget. Parent single-node evidence is independently complete. The internal four-node positional reference is not an asserted Extropic hardware architecture.', '',
        '| Single T | Word T | Single accuracy (%) | Word accuracy (%) | Single minus word (pp) | Descriptive95%interval (pp) |',
        '|---:|---:|---:|---:|---:|---|']
    for bi, (ts, tw) in enumerate([(64, 16), (256, 64), (1024, 256)]):
        single, word = base['values'][bi].mean((0, 1)), arrays['values'][bi].mean((0, 1))
        c = next(c for c in s['contrasts'] if c['budget_index'] == bi and c['contrast'] == 'word_minus_single')
        lo, hi = c['descriptive_95_interval'][0]
        lines.append(f'|{ts}|{tw}|{100*single[0]:.4f}|{100*word[0]:.4f}|{100*(single[0]-word[0]):+.4f}|[{100*-hi:+.6f},{100*-lo:+.6f}]|')
    lines += ['', 'The ranking crosses over: word is stronger at the two smaller retained-update budgets; at the largest, single53.5644%exceedsword51.7068%by1.8576points, descriptive95%interval[0.027990,3.681615]points. All three budgets were fixed beforehand and are retained. These unadjusted descriptive intervals across five model units are not fresh generalization, equivalence, a newly chosen success gate or a universal superiority claim.', '',
        'Both frozen representations miss the approximate1–2point target relative toFP3256.2580%: largest-budget single loses2.6936points and word4.5512points. The moderate baseline/compact16x16RGBarchitecture and consumed-data limitation remain explicit. Probability-aware training/full task confirmation is a separate queued study; it does not retroactively rescue this frozen comparison.', '',
        '| Word T | Word minus reference | Accuracy difference (pp) | Descriptive95%interval (pp) | CE difference | Descriptive95%CE interval |',
        '|---:|---|---:|---|---:|---|']
    for bi, name, point, interval in contrasts:
        lines.append(f"|{[16,64,256][bi]}|{name}|{100*point[0]:+.4f}|[{100*interval[0,0]:+.6f},{100*interval[0,1]:+.6f}]|{point[1]:+.8f}|[{interval[1,0]:+.8f},{interval[1,1]:+.8f}]|")
    lines += ['', 'All9237375000integer counters/15million final logits were regenerated from public states; all150case metrics/500000mean-control logits/19640mean-weight coordinates/120000bootstrap values checked. Verification trajectories are not extra quality replicas. A nominal nonlinear mean-weight control is not E[finite-sample CNN].', '',
        'At each pair3928single versus15712word weight nodes andTs=4Tw match retained scheduled evaluations; each full production/public pass329952000000retained updates. Word warmup2946000000versus736500000single is extra. The single encoding uses four times the local sample rounds; actual hardware latency is unmeasured. Logical command/metadata payload4656single versus2692word bytes, dynamic states491versus1964bytes, and197064bytes/bank int32counters stay separate. No fourfold total memory/area improvement follows from weight-node counts.', '',
        'Equal scheduled updates do not establish equal physical state changes, energy, controller/retention cost or latency. The separate nominal-IID activity diagnostic does not measure physical energy. This result supports a bounded spatial/temporal quality crossover, with remaining systems costs unresolved.', '',
        '[All statistics](cifar_cnn_native/word_cohort_evidence/statistics.json) · [Public audit](cifar_cnn_native/word_cohort_evidence/native_audit.json) · [Independent statistics](cifar_cnn_native/word_cohort_evidence/statistics_audit.json) · [Resource ledger](cifar_cnn_native/word_cohort_evidence/resource_ledger.json) · [Frozen single-node review](CIFAR_CNN_NATIVE_REVIEW.md). The broad research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'all_contrast_units_checked': 12,
              'bootstrap_scalar_values_unit_checked': 120000,
              'artifact_sha256': {str(path): sha(path) for path in [OUT/'completion.json', OUT/'pipeline.json', OUT/'statistics.json', report]},
              'scope': 'Root source/input/completion/stage/contrast-unit review; bounded frozen-model crossover, not physical systems win or overall goal completion.'}
    (OUT/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (OUT/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(OUT/'root_review.json'): sha(OUT/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('All150RGBwordcases/12contrasts/unit directions/terminal stage receipts reviewed', flush=True)


if __name__ == '__main__':
    main()
