"""Close RGB conventional preparation and native witness without test claims."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard

ROOT = Path('results/thrml_spw/cifar_cnn_native')
TRAIN = ROOT/'training'
WITNESS = ROOT/'native_witness_evidence'


def main():
    assert not (ROOT/'preparation_completion.json').exists()
    for folder in [TRAIN, WITNESS]:
        for path, expected in json.loads((folder/'completion.json').read_text())['artifact_sha256'].items():
            assert sha(path) == expected, path
        guard(json.loads((folder/'protocol.json').read_text()))
        assert json.loads((folder/'evidence_audit.json').read_text())['passed']
    pipeline = json.loads((TRAIN/'pipeline.json').read_text())
    assert pipeline['status'] == 'all5conventionalRGBcheckpoints_validation_audited_native_pending_goal_active'
    for stage in pipeline['stages']:
        assert stage['status'] == 'completed' and stage['returncode'] == 0
        assert sha(TRAIN/'logs'/(stage['stage']+'.log')) == stage['log_sha256']
    training = json.loads((TRAIN/'manifest.json').read_text())
    assert training['complete'] and len(training['rows']) == 5 and not training['test_scored']
    for row in training['rows']:
        with np.load(row['path']) as a:
            assert a['conv1_weight'].shape == (8, 3, 3, 3)
            assert a['conv2_weight'].shape == (16, 8, 3, 3)
            assert a['head_weight'].shape == (10, 256)
            assert sum(a[k].size for k in ['conv1_weight', 'conv2_weight', 'head_weight']) == 3928
            assert a['reference_validation_logits'].shape == (5000, 10)
            assert np.isfinite(a['reference_validation_logits']).all()
    witness = json.loads((WITNESS/'evidence_audit.json').read_text())
    assert len(witness['rows']) == 15 and witness['counter_entries_exact'] == 1477980 and witness['all420canonical_arrays_exact']
    audit = json.loads((TRAIN/'evidence_audit.json').read_text())
    assert audit['validation_logit_values_checked'] == 250000 and audit['permutations_checked'] == 100
    check = json.loads((ROOT/'preparation_features_check.json').read_text())
    assert check['passed'] and check['input_coordinates_exact'] == 49152
    for file, expected in check['source_sha256'].items():
        assert sha(file) == expected, file
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = ROOT.parent/'CIFAR_CNN_NATIVE_PREPARATION_REPORT.md'
    assert not report.exists()
    lines = ['# Compact RGB CIFAR-10 CNN: conventional preparation and native witness', '',
             f'Recorded {now}. Five independently initialized conventional CNNs have completed fixed training and full validation-export checks. All five also pass the small-count official-THRML CPU native/public witness. No CIFAR-10 test score or full native task result is established by this preparation.', '',
             'The new controlled architecture uses 16×16 RGB input, Conv3→8/ReLU/maxpool, Conv8→16/ReLU/maxpool and a 256→10 head: 3,928 signed weights and 34 deterministic biases. It is smaller than the old single-seed software CIFAR pilot and is not a reproduction of that architecture. The moderate validation accuracy limits the scope of any subsequent result; this is not a modern competitive CIFAR classifier.', '',
             'The cached training corpus is split deterministically into 45,000 training and 5,000 validation images. Each of five seeds trains for exactly 20 Adam epochs, batch256, lr0.003 with cosine decay to0.0003. No augmentation or fitted normalization is used. A 2×2 average pool follows FP32 raw/255 RGB conversion. Lowest full validation CE selects each checkpoint; all five seeds and all epochs remain retained. The initializations, all 100 minibatch permutations, selected states, checkpoint/NPZ exports, all 250,000 selected validation logits and validation-only activation spans are checked. This is not an independent full retraining.', '',
             '| Seed | Selected epoch | Validation accuracy (%) | Validation CE |', '|---:|---:|---:|---:|']
    for row in audit['rows']:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | {100*row['validation_accuracy']:.2f} | {row['validation_CE']:.8f} |")
    lines += ['', 'Selected post-ReLU/maxpool activation maxima on the full validation set define spans with a fixed 1.25 multiplier; input span is one. No test activation fitting or test-label checkpoint selection occurs. CIFAR test examples were consumed by earlier pilots, so future tests must not be described as wholly fresh generalization confirmation.', '',
              'The first 64 training images provide an additional preprocessing witness: all 49,152 RGB coordinates from Torch average pooling match a separate NumPy corner-sum calculation exactly. Full validation preprocessing is independently checked as well.', '',
              'For the native witness, every selected checkpoint runs on the first 16 validation images, two eight-image banks, at T1/4/16. Each signed weight maps to one free normalized-probability p-bit plus fixed per-output range metadata and finite eight-bit fields over ±4. ACT8 inputs and hidden activations are deterministic; integer counters accumulate before external F64 signed-range readout/ReLU/maxpool. All 15 cases, 420 canonical arrays and 1,477,980 counters match independently constructed public raw-state recovery exactly. The native and replay passes each execute 824,880 retained-sample node updates plus 117,840 warmup node updates.', '',
              'The new shapes need 49,266 int32 output counters per eight-image bank, or 197,064 bytes, apart from input/readout buffers. A T1024 head counter is bounded by255×256×1024=66,846,720, below signed int32 overflow. These are logical/software counts, not physical area/energy/latency. Bias/control precision, range metadata and all other buffers still matter.', '',
              'Full-budget GPU validation, source-frozen deterministic controls and full native test/statistical comparisons remain required. Small-count CPU correctness does not establish task accuracy, sampling efficiency, a hardware benefit or an Extropic disproof. The full long-horizon research goal remains active.', '',
              '[Training protocol](cifar_cnn_native/training/protocol.json), [all five checkpoint exports](cifar_cnn_native/training/manifest.json), [full validation audit](cifar_cnn_native/training/evidence_audit.json), [training completion](cifar_cnn_native/training/completion.json), [native witness protocol](cifar_cnn_native/native_witness_evidence/protocol.json), [native/public evidence](cifar_cnn_native/native_witness_evidence/evidence_audit.json), [native witness completion](cifar_cnn_native/native_witness_evidence/completion.json).']
    report.write_text('\n'.join(lines)+'\n')
    write(ROOT/'preparation_review.json', {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
                                        'all_training_witness_completion_and_source_input_hashes_checked': True,
                                        'all5checkpoint_shapes_and_source_payloads_checked': True,
                                        'report_sha256': sha(report), 'test_scored': False})
    paths = [report, ROOT/'preparation_review.json', ROOT/'preparation_features_check.json', TRAIN/'completion.json', WITNESS/'completion.json', TRAIN/'pipeline.json']
    write(ROOT/'preparation_completion.json', {'recorded_utc': now, 'status': 'RGBtraining_and_small_native_witness_reviewed_fulltask_pending_goal_active',
                                             'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All5RGBconventional checkpoints/native witness reviewed; preparation report published', flush=True)


if __name__ == '__main__':
    main()
