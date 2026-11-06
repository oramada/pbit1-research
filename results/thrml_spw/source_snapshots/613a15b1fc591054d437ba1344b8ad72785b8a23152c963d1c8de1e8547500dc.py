"""Review complete Gaussian controls and publish every predeclared contrast."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.gaussian_control.common import BASE, OUT as CONTROL, MODES, sha, write, guard
from results.thrml_spw.fashion_cnn_native.gaussian_statistics.study import OUT as STATS


def main():
    assert not (STATS/'review_completion.json').exists()
    completion = json.loads((STATS/'completion.json').read_text())
    for path, expected in completion['artifact_sha256'].items():
        assert sha(path) == expected, path
    for folder in [CONTROL, STATS]:
        guard(json.loads((folder/'protocol.json').read_text()))
    pipeline = json.loads((CONTROL/'pipeline.json').read_text())
    assert pipeline['status'] == 'all300Gaussian_cases_numerically_audited_statistics_pending_goal_active'
    for stage in pipeline['stages']:
        assert stage['status'] == 'completed' and stage['returncode'] == 0
        assert sha(CONTROL/'logs'/(stage['stage']+'.log')) == stage['log_sha256']
    manifest = json.loads((CONTROL/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['rows']) == 300
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as arrays:
            assert arrays['logits'].shape == (10000, 10) and arrays['logits'].dtype == np.float64
            assert np.isfinite(arrays['logits']).all()
            assert arrays['clipped_coordinates'].shape == (3,) and arrays['clipped_coordinates'].dtype == np.int64
            assert np.all(arrays['clipped_coordinates'] >= 0)
    audit = json.loads((STATS/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['bootstrap_scalar_values_checked'] == 150000
    data = json.loads((STATS/'statistics.json').read_text())
    compared = [r for r in data['contrasts'] if r['contrast_index'] in [0, 1]]
    assert len(compared) == 6
    assert all(low <= 0 <= high for r in compared for low, high in r['descriptive_95_interval'])
    old = json.loads((BASE/'statistics.json').read_text())
    native = {r['T']: r['mean'] for r in old['summaries']}
    values = {(r['T'], r['mode']): r['mean'] for r in data['summaries']}
    numeric = json.loads((CONTROL/'numerical_audit.json').read_text())
    worst = max(r['maximum_logit_error'] for r in numeric['rows'])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = CONTROL.parent.parent/'FASHION_CNN_GAUSSIAN_CONTROL_REPORT.md'
    assert not report.exists()
    lines = ['# Frozen CNN: variance-matched Gaussian negative control', '',
             f'Recorded {now}. All 300 full evaluation cases, 30 million logits, three million per-example losses and 150,000 bootstrap scalar values have passed their respective numerical/statistical checks.', '',
             'Variance-matched Gaussian weight noise produces similar point accuracy and loss to the native single-p-bit CNN at the three tested sampling budgets. All six Gaussian-minus-native accuracy and loss difference intervals include zero. This does not establish equivalence or prove that Bernoulli and Gaussian noise are interchangeable outside this setting. The comparison provides no evidence of a task-quality advantage specific to the Bernoulli distribution here; the stationary-probability representation and possible physical-device benefits remain separate questions.', '',
             '| Samples T | Native accuracy | Gaussian, original-weight center | Gaussian, programmed-mean center |',
             '|---:|---:|---:|---:|']
    for t in [64, 256, 1024]:
        lines.append(f'| {t} | {100*native[t][0]:.4f}% | {100*values[t, MODES[0]][0]:.4f}% | {100*values[t, MODES[1]][0]:.4f}% |')
    lines += ['', '| Samples T | Native mean CE | Gaussian, original-weight center | Gaussian, programmed-mean center |',
              '|---:|---:|---:|---:|']
    for t in [64, 256, 1024]:
        lines.append(f'| {t} | {native[t][1]:.8f} | {values[t, MODES[0]][1]:.8f} | {values[t, MODES[1]][1]:.8f} |')
    lines += ['', 'FP32_F64 accuracy is 85.0860% and mean CE is 0.41493269. Even the original-weight-centered Gaussian control loses 1.9432 accuracy points at T1024. Thus removing finite-control mean bias alone does not eliminate the degradation at this variance level. This is a bounded diagnostic, not a demonstration of the full causal decomposition of neural error.', '',
              'Each condition uses the same five frozen original CNNs, all 10,000 historically consumed Fashion-MNIST images and ten new Gaussian Monte Carlo seeds. These are not the fine-tuned models whose native replay remains a separate job. Gaussian variance for a weight is 4r²p(1−p)/T, where p comes from the original finite eight-bit field command. Mode 0 adds this noise to the original FP32 weight; mode 1 adds it to the nominal programmed mean. Gaussian tails are unbounded and are not clipped.', '',
              'One perturbed kernel/head is shared across all eight images and all spatial positions in a bank. Independent bank/layer draws preserve the original sharing structure. The two centering modes use identical standard-normal variates. Original deterministic biases, ACT8 range/rounding, F64 arithmetic, ReLU and pooling remain fixed. The normal draws and their contraction are conventional CPU operations, not native THRML calls. T specifies the matched ideal independent temporal-mean variance; it is not a measured physical sample-rate/energy equivalence.', '',
              'All 15 predeclared contrasts follow. Accuracy differences and intervals are in percentage points; CE differences and intervals are in nats. Each interval is a descriptive 95% hierarchical bootstrap interval over five paired model IDs and ten Monte Carlo seeds, with the images fixed. Native and Gaussian seed indices are resampled independently; the two Gaussian modes share seed indices. There is no image resampling or multiplicity correction, no fresh-data generalization confirmation, no selected favorable budget and no equivalence test.', '',
              '| T | Contrast | Accuracy difference [interval] | CE difference [interval] |',
              '|---:|---|---:|---:|']
    for row in data['contrasts']:
        difference, interval = row['difference'], row['descriptive_95_interval']
        lines.append(f"| {row['T']} | {row['contrast']} | {100*difference[0]:+.4f} [{100*interval[0][0]:+.4f}, {100*interval[0][1]:+.4f}] | {difference[1]:+.8f} [{interval[1][0]:+.8f}, {interval[1][1]:+.8f}] |")
    lines += ['', f'The preflight checks all 13,320 starting-weight variances via separate scalar two-outcome calculations and all 60 validation paths. The full separate NumPy spatial-window implementation checks every saved Torch grouped-convolution logit within rtol2e-12/atol2e-11; maximum observed absolute difference is {worst:.16g}, with all predictions identical. Both implementations intentionally use the same specified Gaussian-weight/RNG helper; this does not independently certify that RNG. Scalar per-example metric calculations and unit-by-unit bootstrap accumulation separately verify all cases/contrasts/summaries.', '',
              'The negative control adds zero native p-bit updates. Native values are taken from the previously completed and independently replayed original CNN cohort. Conventional Gaussian runtime is not compared with native simulator runtime as a hardware speed measurement. There are no new physical energy, latency, memory, area or calibration measurements, and these results do not disprove Extropic.', '',
              '[Full simulation protocol](fashion_cnn_native/gaussian_control_evidence/protocol.json), [preflight](fashion_cnn_native/gaussian_control_evidence/preflight.json), [all retained cases](fashion_cnn_native/gaussian_control_evidence/manifest.json), [full numerical replay](fashion_cnn_native/gaussian_control_evidence/numerical_audit.json), [statistics protocol](fashion_cnn_native/gaussian_statistics_evidence/protocol.json), [all results](fashion_cnn_native/gaussian_statistics_evidence/statistics.json), [statistical verification](fashion_cnn_native/gaussian_statistics_evidence/evidence_audit.json), [completion](fashion_cnn_native/gaussian_statistics_evidence/completion.json).', '',
              'Reproduction uses the frozen gaussian_control study prepare command, its pipeline, then the separately frozen gaussian_statistics study prepare/run commands and verifier. Use an isolated worktree with fresh output directories: original protocols/results refuse overwrites. Sources are archived by SHA256 under source_snapshots. This closes the declared Gaussian negative control for the frozen CNN and leaves the wider research goal active.']
    report.write_text('\n'.join(lines)+'\n')
    review = {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__), 'all300case_hashes_and_schemas_checked': True,
              'all_source_input_completion_and_stage_log_hashes_checked': True, 'new_native_updates': 0,
              'report_sha256': sha(report), 'completion_sha256': sha(STATS/'completion.json')}
    write(STATS/'root_review.json', review)
    paths = [report, STATS/'root_review.json', STATS/'completion.json', CONTROL/'pipeline.json']
    write(STATS/'review_completion.json', {'recorded_utc': now, 'status': 'Gaussian_negative_control_reviewed_goal_active',
                                          'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All Gaussian control artifacts reviewed; report published', flush=True)


if __name__ == '__main__':
    main()
