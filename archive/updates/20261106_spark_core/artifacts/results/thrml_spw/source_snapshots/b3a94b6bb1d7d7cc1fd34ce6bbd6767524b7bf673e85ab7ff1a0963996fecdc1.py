"""Review closed-cohort exports and publish the scientific figure caption."""
import datetime
import json
from pathlib import Path
from results.thrml_spw.fashion_cnn_native.finetune_figures.common import OUT, sha, write, guard


def main():
    assert not (OUT/'review_completion.json').exists()
    for path, expected in json.loads((OUT/'completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == expected, path
    guard(json.loads((OUT/'protocol.json').read_text()))
    check = json.loads((OUT/'evidence_audit.json').read_text())
    assert check['passed'] and check['points_checked'] == 12 and check['bootstrap_scalar_values_checked'] == 120000
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = OUT.parent.parent/'FASHION_CNN_FINETUNING_FIGURE.md'
    assert not report.exists()
    report.write_text('''# Native CNN sampling curves after probability-aware fine-tuning

The complete curves show how conventional probability-aware fine-tuning changes subsequent native THRML inference on the five-model Fashion-MNIST CNN cohort. All three budgets and all four method conditions are retained; all 12 points and 120,000 bootstrap scalar values have passed separate numerical checks. No new model evaluation or quality replication is added by plotting existing evidence.

![Native CNN accuracy and cross-entropy versus retained-sample updates](fashion_cnn_native/finetune_figure_evidence/native_cnn_finetuning.png)

Each condition has five original model IDs and ten inference seeds. Error bars are absolute 95% descriptive hierarchical bootstrap intervals, resampling model units and within-model inference seeds with the indices retained by the completed experiments. Evaluation images are fixed and historically consumed. Reference lines show five-model means: FP32_F64 uses the original FP32 stored weights with F64 arithmetic and unquantized activations; INT4 min–max uses dequantized arithmetic and ACT8. They are not optimized hardware kernels.

For one-p-bit conditions, T is 64, 256 or 1024. For the internal four-node reference, T is 16, 64 or 256, so K×T matches retained-sample node evaluations. The x-axis excludes the separately charged warmup, training and independent replay work. Every plotted native condition uses integer accumulation before external F64 signed-range readout and the original nonlinearities. Both training methods retain the original per-output centers/radii and biases; they use the predeclared exported F32 activation spans.

Expected-weight and sample-aware training are conventional CPU procedures. The sampled method uses the declared binomial-mean straight-through surrogate; this is not a claim of native or unbiased stochastic gradient training. Compared with the frozen methods, the trained curves include extra optimization. The plot therefore cannot isolate representation effects under matched training. Training changes expected weights as well as variance; these curves do not isolate a causal mechanism for the gain.

Sample-aware one-p-bit inference reaches 84.0760% at T256 and 85.0262% at T1024, against the original FP32_F64 mean of 85.0860%. The plotted absolute intervals are not equivalence tests or confidence intervals of a paired difference. All paired contrast intervals are retained in the full cohort report. The original frozen-model target miss is unchanged.

One-node methods use 2,664 free weight nodes and a nominal logical parameter payload of 3,392 bytes/model. The internal four-node reference uses 10,656 free nodes and 2,060 bytes of logical commands/metadata. Dynamic states, controller/readout circuits, counters, activation buffers and training/evidence storage are distinct. These curves establish no measured physical memory, area, energy or latency advantage and are not a full systems Pareto frontier.

Download [SVG](fashion_cnn_native/finetune_figure_evidence/native_cnn_finetuning.svg), [PNG](fashion_cnn_native/finetune_figure_evidence/native_cnn_finetuning.png) or [all plotted data and work counts as CSV](fashion_cnn_native/finetune_figure_evidence/sampling_curves.csv). [Figure protocol](fashion_cnn_native/finetune_figure_evidence/protocol.json), [all data](fashion_cnn_native/finetune_figure_evidence/data.json), [independent figure audit](fashion_cnn_native/finetune_figure_evidence/evidence_audit.json), [figure completion](fashion_cnn_native/finetune_figure_evidence/completion.json), [full fine-tuned cohort and all contrasts](FASHION_CNN_PROBABILITY_FINETUNE_REVIEW.md).

The PNG was visually inspected for readable labels, legends, uncertainty bars and the complete visible endpoints. The broader research goal remains active; the separate exact-budget sample-allocation experiment is still running.
''')
    write(OUT/'root_review.json', {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
                                 'all_source_input_and_completion_hashes_checked': True,
                                 'visual_inspection': 'PNG viewed: axes/labels/legend/intervals and scope footnotes legible, no plot content cropped.',
                                 'report_sha256': sha(report), 'no_new_model_evaluations': True})
    paths = [report, OUT/'root_review.json', OUT/'completion.json']
    write(OUT/'review_completion.json', {'recorded_utc': now, 'status': 'fine_tuned_CNN_figure_reviewed_goal_active',
                                       'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All fine-tuning figure exports reviewed; report published', flush=True)


if __name__ == '__main__':
    main()
