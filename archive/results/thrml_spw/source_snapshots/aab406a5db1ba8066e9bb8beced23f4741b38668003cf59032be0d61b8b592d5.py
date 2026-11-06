"""Review terminal T4096 evidence without changing any sealed experiment file."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_high_budget_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'review_completion.json').exists()
    source = sha(__file__)
    assert (ROOT.parent/'source_snapshots'/f'{source}.py').read_bytes() == Path(__file__).read_bytes()
    completion = json.loads((OUT/'completion.json').read_text())
    terminal = 'all100_T4096_RGB_cases_independently_audited_goal_active'
    assert completion['status'] == terminal
    for path, digest in completion['artifact_sha256'].items():
        assert sha(path) == digest, path
    protocol = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in protocol[group].items():
            assert sha(path) == digest, path
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    assert pipeline['status'] == terminal
    assert [r['stage'] for r in pipeline['stages']] == [
        'controls_copy', 'production', 'public_replay', 'statistics', 'independent_statistics']
    for row in pipeline['stages']:
        assert row['status'] == 'completed' and row['returncode'] == 0
        assert sha(OUT/'logs'/(row['stage']+'.log')) == row['log_sha256']
    manifest = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    verified = json.loads((OUT/'statistics_audit.json').read_text())
    assert manifest['complete'] and manifest['independently_audited'] is False
    assert audit['passed'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    assert len(manifest['rows']) == len(audit['rows']) == 100
    assert audit['counter_entries_exact'] == 6158250000
    assert audit['public_updates'] == manifest['native_updates'] == 2011136000000
    assert [r['name'] for r in manifest['rows']] == [r['name'] for r in audit['rows']]
    for row, proof in zip(manifest['rows'], audit['rows']):
        assert sha(row['path']) == row['sha256'] == proof['case_sha256']
        assert proof['all17500_array_receipts_exact'] and proof['all100000_final_logits_exact']
    expected_audit = {'case_metrics_checked': 100, 'native_logits_checked': 10000000,
        'BF16_paths_checked': 10, 'BF16_logits_checked': 1000000,
        'contrast_rows_checked': 15, 'bootstrap_scalar_values_checked': 150000}
    assert verified['passed'] and verified['statistics_sha256'] == sha(OUT/'statistics.json')
    for key, count in expected_audit.items():
        assert verified[key] == count
    stats = json.loads((OUT/'statistics.json').read_text())
    arrays = np.load(OUT/'statistical_arrays.npz')
    parent = np.load(ROOT/'probability_cohort_evidence/statistical_arrays.npz')
    base = np.load(ROOT/'cohort_evidence/statistical_arrays.npz')
    word = np.load(ROOT/'word_cohort_evidence/statistical_arrays.npz')
    precision = np.load(ROOT/'precision_evidence/statistical_arrays.npz')
    values = arrays['values']
    assert values.shape == (2, 5, 10, 2)
    assert len(stats['summaries']) == 2 and len(stats['contrasts']) == 15
    expected = {(method, reference) for method, reference in protocol['contrasts']}
    seen = set()
    for contrast in stats['contrasts']:
        method, name = contrast['method_index'], contrast['reference']
        assert (method, name) not in seen and contrast['T'] == 4096
        seen.add((method, name))
        reference = {
            'BF16_weights_F64': precision['control_metrics'][1],
            'BF16_Torch_CPU': precision['control_metrics'][3],
            'FP32_F64': base['baseline_metrics'][0],
            'own_mean': parent['mean_metrics'][method],
            'own_T1024': parent['values'][method, 2],
            'original_T1024': base['values'][2],
            'word_T256': word['values'][2],
            'expected_T4096': values[0]}[name]
        point = values[method].mean((0, 1))-reference.mean(tuple(range(reference.ndim-1)))
        np.testing.assert_allclose(point, contrast['difference'], rtol=2e-9, atol=3e-12)
        distribution = arrays[contrast['array_key']]
        assert distribution.shape == (5000, 2)
        np.testing.assert_allclose(np.quantile(distribution, [.025, .975], axis=0).T,
                                   contrast['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert seen == expected and len(seen) == 15
    for row in stats['summaries']:
        method = row['method_index']
        np.testing.assert_allclose(values[method].mean((0, 1)), row['mean'], rtol=2e-12, atol=2e-13)
        np.testing.assert_array_equal(row['clipped_coordinates'], np.sum([
            r['clipped_coordinates'] for r in manifest['rows'] if r['method'] == method], axis=0))
    ledger = json.loads((OUT/'resource_ledger.json').read_text())
    assert ledger['weight_pbits'] == 3928 and ledger['deterministic_biases'] == 34
    assert ledger['retained_updates_per_native_or_public_pass'] == 2011136000000
    assert ledger['warmup_updates_per_native_or_public_pass'] == 491000000
    assert ledger['T4096_to_T1024_local_round_ratio'] == 4
    report = ROOT.parent/'CIFAR_CNN_HIGH_BUDGET_REVIEW.md'
    assert not report.exists()
    lines = ['# Reviewed T4096 single-p-bit RGB results', '',
        'All 100 native cases, complete public replay, all five successful pipeline stages/log hashes, frozen source/input hashes and completion-bound files pass this terminal review. The independent verifier recomputed all 100 task metrics from 10 million logits, 10 BF16 control paths from 1 million logits, and all 150000 bootstrap scalar values. This review checks all 15 contrast directions, units and intervals against the saved arrays.', '',
        '| Method | Accuracy (%) | Difference from FP32 (pp) | Difference from CPU BF16 (pp) | Descriptive 95% BF16 interval (pp) |',
        '|---|---:|---:|---:|---|']
    for row in stats['summaries']:
        contrasts = {c['reference']: c for c in stats['contrasts'] if c['method_index'] == row['method_index']}
        bf = contrasts['BF16_Torch_CPU']
        lo, hi = bf['descriptive_95_interval'][0]
        lines.append(f"|{row['method']}|{100*row['mean'][0]:.4f}|{100*contrasts['FP32_F64']['difference'][0]:+.4f}|{100*bf['difference'][0]:+.4f}|[{100*lo:+.6f}, {100*hi:+.6f}]|")
    lines += ['',
        'Expected-weight fine-tuning reaches 56.5842%, compared with 56.2580% original FP32 and 56.2940% CPU BF16. Its observed BF16 difference is +0.2902 percentage points; the descriptive interval crosses zero. The sample-aware method reaches 55.6122%, a BF16 difference of -0.6818 points, and remains worse than the expected-weight method at this budget. Both results and all lower-budget negative results remain reported.', '',
        'This supports useful one-free-p-bit-per-weight inference at comparable observed task accuracy in this compact CNN. It does not establish per-parameter BF16 numerical precision, formal statistical equivalence or a physical systems advantage. There are five paired starting models and ten inference seeds per method; the 10000 test images are historically consumed. T4096 was selected after earlier results, both converted methods have eight extra training epochs versus the frozen FP32/BF16 controls, and no confirmatory equivalence margin or fresh test set was selected.', '',
        'Sampling is four times the local rounds of T1024. Each production/replay pass uses 2011136000000 retained plus 491000000 warmup scheduled node evaluations. One free stochastic weight node per normalized parameter uses 3928 weight nodes; 34 deterministic biases, range metadata, 8-bit commands, ACT8 activations, integer output counters and external F64 readout remain separate resources. Word-T256 comparisons here use less sampling work and are not matched-resource advantages. Scheduled evaluations are not measured bit flips, energy, area or hardware latency.', '',
        'Expected-weight cases record 19 clipped coordinates in the third-layer aggregate; the sample-aware cases record none. The frozen raw-case storage estimate omitted 2400 bytes of clipping arrays, as recorded in the earlier production review; the existing output bounds cover these bytes and the protocol remains unchanged.', '',
        '[Full results](CIFAR_CNN_HIGH_BUDGET_REPORT.md) · [All statistics](cifar_cnn_native/probability_high_budget_evidence/statistics.json) · [Completion](cifar_cnn_native/probability_high_budget_evidence/completion.json)', '',
        'The broader research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': source, 'contrast_rows_unit_checked': 15,
        'bootstrap_scalar_values_interval_checked': 150000,
        'artifact_sha256': {str(path): sha(path) for path in [OUT/'completion.json', OUT/'pipeline.json', report]},
        'scope': 'Terminal completion/source/input/log/contrast/resource review; no new inference replicas, numerical BF16 equivalence, physical advantage or full goal completion.'}
    (OUT/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (OUT/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(OUT/'root_review.json'): sha(OUT/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('T4096 cohort closed: 100 cases, five stages, 15 contrasts; broader goal active')


if __name__ == '__main__':
    main()
