"""Review validation schedules and CPU native witness without test-quality claims."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, sha, write, guard


def main():
    validation = ROOT/'layer_allocation_evidence'
    witness = ROOT/'allocated_witness_evidence'
    queued = ROOT/'allocated_native_evidence'
    assert not (validation/'review_completion.json').exists()
    for folder in [validation, witness]:
        for path, expected in json.loads((folder/'completion.json').read_text())['artifact_sha256'].items():
            assert sha(path) == expected, path
        guard(json.loads((folder/'protocol.json').read_text()))
    guard(json.loads((queued/'protocol.json').read_text()))
    stages = json.loads((validation/'pipeline.json').read_text())
    assert stages['status'] == 'validation_allocation_independently_closed_native_test_pending_goal_active'
    for stage in stages['stages']:
        assert stage['status'] == 'completed' and stage['returncode'] == 0
        assert sha(validation/'logs'/(stage['stage']+'.log')) == stage['log_sha256']
    manifest = json.loads((validation/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['rows']) == 150 and len(manifest['baseline_rows']) == 5
    for row in manifest['rows']+manifest['baseline_rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as a:
            assert a['logits'].shape == (5000, 10) and a['logits'].dtype == np.float64
            assert np.isfinite(a['logits']).all()
    selection = json.loads((validation/'allocation.json').read_text())
    assert json.loads((validation/'selection_audit.json').read_text())['passed']
    check = json.loads((witness/'evidence_audit.json').read_text())
    assert check['passed'] and check['counter_entries_exact'] == 566190 and len(check['rows']) == 15
    launch = json.loads((queued/'launch.json').read_text())
    proc = Path('/proc')/str(launch['pid'])
    stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
    assert int(stat[19]) == launch['startticks'] and stat[0] != 'Z'
    assert b'allocated_native.pipeline' in proc.joinpath('cmdline').read_bytes()
    state = json.loads((queued/'pipeline.json').read_text())
    assert state['status'] == 'waiting_for_live_finetuned_pipeline'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = ROOT.parent/'FASHION_CNN_LAYER_ALLOCATION_PREPARATION_REPORT.md'
    assert not report.exists()
    lines = ['# CNN layer sampling allocation: validation preparation', '',
             f'Recorded {now}. All 150 validation sensitivity cases and five nominal baselines have passed numerical/scalar-loss checks. Five model-specific schedules are frozen and independently verified as global optima of the declared discrete validation surrogate. A basic CPU native/public witness has passed; the full GPU native comparison is queued and has no task-quality result yet.', '',
             'Each sensitivity case adds variance-matched Gaussian noise at T256 to just one layer, keeping the other two layers at their finite-control nominal means. All 5,000 training-validation images, five original frozen CNNs and ten seeds per layer are retained. This uses neither test images nor test labels for schedule selection. The validation images were already used for checkpoint selection; this is not fresh validation-data confirmation. The Gaussian sensitivity proxy is conventional computation, not native p-bit sampling.', '',
             'The coefficient is S_l=max(mean isolated-layer validation CE−nominal-mean validation CE, 1e−4). The fixed objective minimizes sum_l S_l·256/T_l across all integer sample budgets from 64 through 1024, subject to 72T_conv1+1152T_conv2+1440T_head=2664·256. This gives 20,009 feasible schedules per model. Separate scalar enumeration, with a different traversal from the vectorized selector, verifies the global optimum, tie-breaking and exact cost for each model. A separable inverse-T loss proxy need not capture neural layer interactions.', '',
             '| Model | Conv1 samples | Conv2 samples | Head samples | Uniform proxy | Allocated proxy |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in selection['rows']:
        a, b, c = row['samples']
        lines.append(f"| {row['model']} | {a} | {b} | {c} | {row['uniform_surrogate']:.8f} | {row['surrogate']:.8f} |")
    lines += ['', 'All schedules use exactly 681,984 retained-sample weight-node updates per eight-image bank, matching uniform T256. All 2,664 normalized weights remain free p-bits. Each representation needs its original controller/range/bias metadata. A three-entry uint16 schedule adds six bytes per model, versus two bytes for a single uniform uint16 budget; this metadata is not a p-bit or memory-density win.', '',
              'Uniform has 768 summed layer sample rounds and a maximum layer budget of 256. Allocated sums and maxima vary. Equal node evaluations do not imply equal physical latency, energy, readout or accumulation cost. One additional warmup sweep is charged separately per bank/layer: 2,664 node updates per bank, equal across the two conditions. Scheduled node updates are distinct from physical flips.', '',
              'The validation audit independently checks all 7,750,000 logits and 775,000 scalar per-example losses using NumPy spatial contractions and scalar log-sum-exp, versus Torch grouped convolution. Both implementations intentionally share the specified Gaussian draw helper. This checks numerical execution rather than independently certifying the RNG.', '',
              'The CPU official-THRML witness tests every original checkpoint on the first eight validation inputs with schedules (1,2,4), (4,2,1) and (4,4,4). All 15 cases, 210 canonical arrays and 566,190 integer counter entries match separately constructed public raw-state recovery exactly. It executes 114,120 retained-sample node updates in production and the same in replay, plus 39,960 warmup node updates per pass. These small-count CPU cases validate mixed per-layer operator logic, not full-budget GPU behavior or task quality.', '',
              'The queued study waits for the actual current fine-tuned pipeline to close successfully, then runs a GPU preflight on all five models at uniform64/256/1024 and each allocated schedule. Only after all 20 validation preflight cases match public states may the full 100 test cases run: five paired model IDs, ten fresh inference seeds, uniform256 versus frozen allocation, all historically consumed 10,000 images. Each full pass has 85,248,000,000 retained-sample scheduled node updates plus 333,000,000 warmup updates; production and public replay are charged separately. All receipts and final logits must match before statistics. No schedule may be changed based on test outcomes.', '',
              'The full planned statistics retain allocated-minus-uniform and each-minus-FP32_F64, with 5,000 hierarchical model/within-condition-seed bootstrap draws, independent seed resampling across conditions and no image resampling. These will be descriptive fixed-data intervals, not equivalence tests or fresh generalization evidence. No native accuracy improvement, systems advantage or Extropic disproof is established by this preparation.', '',
              '[Validation protocol](fashion_cnn_native/layer_allocation_evidence/protocol.json), [all validation cases](fashion_cnn_native/layer_allocation_evidence/manifest.json), [numerical audit](fashion_cnn_native/layer_allocation_evidence/evidence_audit.json), [frozen schedules](fashion_cnn_native/layer_allocation_evidence/allocation.json), [independent optimum](fashion_cnn_native/layer_allocation_evidence/selection_audit.json), [validation completion](fashion_cnn_native/layer_allocation_evidence/completion.json), [CPU witness](fashion_cnn_native/allocated_witness_evidence/evidence_audit.json), [full native protocol](fashion_cnn_native/allocated_native_evidence/protocol.json), [live native stages](fashion_cnn_native/allocated_native_evidence/pipeline.json).', '',
              'The broader research goal remains active. Full native task quality, statistical confirmation, broader models and physical cost measurements remain open.']
    report.write_text('\n'.join(lines)+'\n')
    write(validation/'root_review.json', {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
                                        'all155validation_case_hashes_and_schemas_checked': True,
                                        'all_source_input_completion_and_log_hashes_checked': True,
                                        'native_queue_pid': launch['pid'], 'native_queue_startticks': launch['startticks'],
                                        'native_test_status': state['status'], 'report_sha256': sha(report)})
    paths = [report, validation/'completion.json', validation/'root_review.json', witness/'completion.json', queued/'protocol.json']
    write(validation/'review_completion.json', {'recorded_utc': now, 'status': 'validation_and_CPU_native_preparation_reviewed_full_native_test_pending_goal_active',
                                               'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('Validation allocation and CPU native witness reviewed; GPU study queued', flush=True)


if __name__ == '__main__':
    main()
