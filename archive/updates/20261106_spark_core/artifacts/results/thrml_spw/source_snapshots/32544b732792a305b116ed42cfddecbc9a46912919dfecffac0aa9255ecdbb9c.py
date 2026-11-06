"""Review frozen RGB protocol and completed full-budget prerequisite gates."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'cohort_evidence'


def main():
    assert not (OUT/'preflight_review_completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    preflight = json.loads((OUT/'preflight.json').read_text())
    assert preflight['passed'] and preflight['protocol_sha256'] == sha(OUT/'protocol.json')
    assert len(preflight['rows']) == 15 and preflight['counter_entries_exact'] == 1477980
    assert preflight['native_updates'] == preflight['public_updates'] == 52792320
    assert {(r['model'], r['T']) for r in preflight['rows']} == {(mi, t) for mi in range(5) for t in [64, 256, 1024]}
    for row in preflight['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as arrays:
            assert len(arrays.files) == 14 and arrays['output_2'].shape == (2, 8, 10)
            for li in range(3):
                assert arrays[f'gate_{li}'].dtype == arrays[f'column_{li}'].dtype == np.int32
    controls = json.loads((OUT/'controls.json').read_text())
    assert controls['complete'] and controls['protocol_sha256'] == sha(OUT/'protocol.json')
    assert controls['test_data_sha256'] == sha(OUT/'test_data.npz')
    assert controls['control_logits_sha256'] == sha(OUT/'control_logits.npz')
    assert len(controls['rows']) == 50 and {(r['model'], r['control']) for r in controls['rows']} == {(mi, name) for mi in range(5) for name in p['controls']}
    selfcheck = json.loads((OUT/'control_selfcheck.json').read_text())
    assert selfcheck['passed'] and len(selfcheck['rows']) == 50
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    for name in ['full_budget_GPU_preflight', 'controls']:
        row = next(r for r in pipeline['stages'] if r['stage'] == name)
        assert row['status'] == 'completed' and row['returncode'] == 0
        assert sha(OUT/'logs'/(name+'.log')) == row['log_sha256']
    launch = json.loads((OUT/'launch.json').read_text())
    proc = Path('/proc')/str(launch['pid'])
    state = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
    assert int(state[19]) == launch['startticks'] and state[0] != 'Z'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = ROOT.parent/'CIFAR_CNN_COHORT_PROTOCOL_REPORT.md'
    assert not report.exists()
    report.write_text('''# Full native compact RGB CIFAR-10 experiment

The 150-case experiment is source-frozen and running: five independently trained compact RGB CNNs, all T64/256/1024 budgets, ten sampling seeds each and all 10,000 standard test images. The moderate 55–57% validation baseline and smaller architecture relative to the old software pilot remain explicit. This is a controlled native encoding benchmark, not a competitive CIFAR classifier.

Before test execution, all 15 GPU validation cases (every checkpoint/budget, first16validation inputs/two banks) matched public raw-state recovery on every counter/activation/mask/output array and final logit: 1,477,980 integer counter entries and 420 canonical arrays checked. Native and public preflight each execute 52,792,320 retained-sample node updates, plus 117,840 separately charged warmup updates. The preceding 50 numerical control paths passed independent Torch-versus-NumPy checks on 64 validation inputs.

All 50 full deterministic controls are saved: FP32_F64, conventional FP32_F32, ACT8 with original weights, INT2/4/8 min–max and symmetric zero-preserving quantization, and the nominal programmed-mean network. Low-bit controls use dequantized arithmetic; they are not hardware kernel performance claims. A nonlinear nominal-mean plug-in is not the expected output of the sampled network. Full control numerical verification remains a required final stage.

Every kernel/head weight uses one free normalized-probability p-bit and finite eight-bit fields over ±4, with per-output center/radius metadata. Biases remain deterministic. Deterministic ACT8 inputs and activations gate integer output counters; averaging occurs before external F64 signed-range readout, ReLU and pooling. Kernel realizations are shared across spatial positions and eight images in each bank. No operational per-weight reconstruction is introduced.

Full production and public replay each require 329,952,000,000 retained-sample node updates plus 736,500,000 warmup updates. Every transient native array must be regenerated and its semantic digest matched, together with all 15 million saved final logits and 9,237,375,000 integer counter entries. Digests cannot decode counters; source/model/data/key/schedule seals and successful regeneration provide the evidence. Verification is not another independent quality replicate.

All three budgets and all 30 deterministic-control contrasts are retained. The fixed statistical protocol uses five paired model units, ten sampling seeds and 5,000 hierarchical model/seed bootstrap draws, with the images held fixed. The final checker independently verifies raw RGB preprocessing and labels, all 50 control paths, all 150 case metrics and 300,000 bootstrap scalar values. These are descriptive intervals on historically consumed test data, not fresh generalization confirmation, equivalence tests or outcome-selected quality gates.

The largest T1024 head counter is bounded by66,846,720, within int32. The software observer uses49,266int32output counters/eight-imagebank,197,064bytes. Logical parameter payload is4,656bytes before separate stochastic states and buffers:3,928eight-bit commands,544bytes ofF64weight ranges,136bytes ofFP32biases and48bytes ofactivation ranges. These are not measured physical memory, area, energy or latency. No Extropic architecture superiority or disproof is claimed.

[Frozen protocol](cifar_cnn_native/cohort_evidence/protocol.json), [50validation control checks](cifar_cnn_native/cohort_evidence/control_selfcheck.json), [full-budget GPU preflight](cifar_cnn_native/cohort_evidence/preflight.json), [preflight integrity review](cifar_cnn_native/cohort_evidence/preflight_root_review.json), [live pipeline](cifar_cnn_native/cohort_evidence/pipeline.json), [five-model training/native CPU preparation](CIFAR_CNN_NATIVE_PREPARATION_REPORT.md).

Native production is underway. Full public replay, task statistics and independent checks remain required before reporting native CIFAR-10 accuracy. The larger research goal remains active.
''')
    write(OUT/'preflight_root_review.json', {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
                                          'all_protocol_source_input_hashes_checked': True,
                                          'all15preflight_case_hashes_and_schemas_checked': True,
                                          'control_metadata_hashes_and_coverage_checked': True,
                                          'actual_parent_pid': launch['pid'], 'actual_parent_startticks': launch['startticks'],
                                          'report_sha256': sha(report), 'scope': 'Prerequisite/source/artifact review only, no labels/accuracy or new draws.'})
    paths = [report, OUT/'protocol.json', OUT/'preflight.json', OUT/'controls.json', OUT/'preflight_root_review.json']
    write(OUT/'preflight_review_completion.json', {'recorded_utc': now, 'status': 'full_RGB_prerequisites_reviewed_native_cohort_running_goal_active',
                                                'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('Full RGB protocol/prerequisite gates reviewed; native production running', flush=True)


if __name__ == '__main__':
    main()
