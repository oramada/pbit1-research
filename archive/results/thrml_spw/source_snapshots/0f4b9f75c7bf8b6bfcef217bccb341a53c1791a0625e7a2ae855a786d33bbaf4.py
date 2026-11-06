"""Review the completed parameter diagnostic and publish its bounded interpretation."""
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    root = Path('results/thrml_spw')
    folder = root/'fashion_cnn_native/probability_mechanism_evidence'
    assert not (folder/'review_completion.json').exists()
    completion = json.loads((folder/'completion.json').read_text())
    for path, expected in completion['artifact_sha256'].items():
        assert sha(path) == expected, path
    protocol = json.loads((folder/'protocol.json').read_text())
    for key in ['source_sha256', 'input_sha256']:
        for path, expected in protocol[key].items():
            assert sha(path) == expected, path
    audit = json.loads((folder/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['weight_conditions_checked'] == 39960
    data = json.loads((folder/'moments.json').read_text())
    assert len(data['layers']) == 45 and len(data['models']) == 15
    summaries = {r['condition']: r for r in data['summaries']}
    conditions = protocol['conditions']
    assert set(summaries) == set(conditions) and all(r['model_units'] == 5 for r in summaries.values())
    now = datetime.datetime.now(datetime.timezone.utc)
    reductions = {}
    for condition in conditions[1:]:
        values = []
        for model in range(5):
            before = next(r for r in data['models'] if r['condition'] == 'starting' and r['model'] == model)
            after = next(r for r in data['models'] if r['condition'] == condition and r['model'] == model)
            values.append(100*(1-after['mean_single_sample_variance']/before['mean_single_sample_variance']))
        reductions[condition] = {'paired_model_percent_reductions': values, 'mean_percent': sum(values)/5}
    report = root/'FASHION_CNN_PROBABILITY_MECHANISM_REPORT.md'
    assert not report.exists()
    lines = ['# Learned probability controls: parameter-only mechanism check', '',
             f'Recorded {now.isoformat()}. All 39,960 nominal weight conditions have passed a separate scalar two-outcome calculation. This descriptive diagnostic uses five trained CNNs, their starting checkpoints and both fine-tuning methods. It uses no images, labels, task scores or new simulator draws.', '',
             'Fine-tuning modestly reduces nominal weight sampling variance. This does not determine whether neural accuracy improves or how many samples inference needs. The full task replay and statistical audits remain separate gates.', '',
             '| Condition | Mean per-sample weight variance | Endpoint command fraction | Probabilities outside (0.01, 0.99) | Mean cell entropy (bits) |',
             '|---|---:|---:|---:|---:|']
    labels = ['Starting', 'Expected-weight training', 'Binomial-mean STE, T256']
    for condition, label in zip(conditions, labels):
        r = summaries[condition]
        lines.append(f"| {label} | {r['mean_single_sample_variance']:.9f} | {100*r['endpoint_command_fraction']:.4f}% | {100*r['probability_extreme_fraction_001']:.4f}% | {r['mean_stationary_cell_entropy_bits']:.6f} |")
    lines += ['', 'Each whole-model value weights actual weight coordinates equally, then averages five model units. There are 2,664 weights per model: 72 in conv1, 1,152 in conv2 and 1,440 in the head. Weight coordinates are not treated as independent experimental replications. No significance test or confidence interval is claimed for this parameter diagnostic.', '',
              'The mean of the five paired-model relative variance reductions is '+f"{reductions[conditions[1]]['mean_percent']:.4f}% for expected-weight training and {reductions[conditions[2]]['mean_percent']:.4f}% for sampled-weight training. These differ slightly from taking the ratio of the two condition means.", '',
              '| Layer | Starting variance | Expected-weight variance | Sampled-weight variance |',
              '|---|---:|---:|---:|']
    for layer in protocol['layers']:
        values = [sum(r['mean_single_sample_variance'] for r in data['layers'] if r['condition'] == c and r['layer'] == layer)/5 for c in conditions]
        lines.append('| '+layer+' | '+' | '.join(f'{v:.9f}' for v in values)+' |')
    lines += ['', 'For a fixed center c, radius r and field h, the nominal law is p=sigmoid(2h), mean=c+r(2p−1), variance=4r²p(1−p), and d(mean)/dh=4rp(1−p). The producer evaluates those formulas. The verifier separately enumerates the two outcomes c−r and c+r, computes their weighted variance, and obtains the field derivative as twice the covariance with the binary state. It checks every retained coordinate, all 45 layer summaries and all 15 model summaries; code histograms and entropy calculations use separate scalar/Counter calculations.', '',
              'Original per-output centers and radii stay fixed. Fine-tuned 8-bit commands are read directly from exported checkpoints and checked against latent-field rounding; expected weights are never min/max refitted. Command endpoints are codes 0 and 255 on the finite ±4 field grid. Endpoint-code fraction falls while the broader extreme-probability fraction rises slightly: these are different saturation measures.', '',
              'Cell entropy measures the uncertainty of a sampled binary state. Across-weight command-frequency entropy instead describes the histogram of programmed settings: its condition means are '+', '.join(f"{summaries[c]['empirical_command_frequency_entropy_bits']:.6f}" for c in conditions)+' bits. Neither statistic establishes a storage saving. The implementation still programs an 8-bit command per weight and stores range metadata.', '',
              'Mean-squared changes from the original FP32 weights are retained in the machine-readable evidence as intentional learned changes, not reconstruction error or a task-quality score. Reduced marginal variance does not establish reduced output variance after nonlinearities, robustness to physical field errors, empirical independence, correlation behavior, accuracy, latency or energy. All three conditions and all five models are retained; methods and sampling budgets are not selected using this analysis.', '',
              '[Protocol](fashion_cnn_native/probability_mechanism_evidence/protocol.json), [all metrics](fashion_cnn_native/probability_mechanism_evidence/moments.json), [coordinate arrays](fashion_cnn_native/probability_mechanism_evidence/vectors.npz), [independent check](fashion_cnn_native/probability_mechanism_evidence/evidence_audit.json), [completion](fashion_cnn_native/probability_mechanism_evidence/completion.json).', '',
              'Reproduce in a fresh output directory by retaining the frozen sources and inputs, then running the study prepare/run commands and the separate verifier. The original directory refuses overwrite after protocol/results/completion creation. The long-horizon goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    write(folder/'root_review.json', {'passed': True, 'recorded_utc': now.isoformat(), 'completion_sha256': sha(folder/'completion.json'),
                                    'source_sha256': sha(__file__), 'all_protocol_sources_and_inputs_match': True,
                                    'all_completion_artifacts_match': True, 'paired_reductions': reductions,
                                    'report_sha256': sha(report), 'new_native_updates': 0})
    paths = [report, folder/'completion.json', folder/'root_review.json']
    write(folder/'review_completion.json', {'recorded_utc': now.isoformat(), 'status': 'parameter_diagnostic_reviewed_goal_active',
                                           'artifact_sha256': {str(p): sha(p) for p in paths}})
    print(json.dumps(reductions, indent=2))


if __name__ == '__main__':
    main()
