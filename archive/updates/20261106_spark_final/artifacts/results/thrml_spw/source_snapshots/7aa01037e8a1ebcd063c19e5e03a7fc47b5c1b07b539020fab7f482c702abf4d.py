"""Full native allocation/statistics closure and bounded scientific report."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, sha, write, guard

NATIVE = ROOT/'allocated_native_evidence'
STATS = ROOT/'allocated_statistics_evidence'


def main():
    assert not (STATS/'review_completion.json').exists()
    for path, expected in json.loads((STATS/'completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == expected, path
    for folder, status in [(NATIVE, 'all100native_allocated_comparisons_publicly_replayed_statistics_pending_goal_active'),
                           (STATS, 'all100matched_budget_CNN_cases_statistically_closed_goal_active')]:
        guard(json.loads((folder/'protocol.json').read_text()))
        pipeline = json.loads((folder/'pipeline.json').read_text())
        assert pipeline['status'] == status
        for stage in pipeline['stages']:
            assert stage['status'] == 'completed' and stage['returncode'] == 0
            assert sha(folder/'logs'/(stage['stage']+'.log')) == stage['log_sha256']
    m = json.loads((NATIVE/'manifest.json').read_text())
    a = json.loads((NATIVE/'native_audit.json').read_text())
    assert m['complete'] and a['complete'] and a['passed'] and a['manifest_sha256'] == sha(NATIVE/'manifest.json')
    assert len(m['rows']) == len(a['rows']) == 100 and a['counter_entries'] == 4718250000
    assert all(r['all_receipts_and_final_logits_exact'] for r in a['rows'])
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as arrays:
            assert arrays['logits'].shape == (10000, 10) and arrays['logits'].dtype == np.float64
            assert np.isfinite(arrays['logits']).all()
            assert arrays['receipts'].shape == (1250, 14, 32) and arrays['receipts'].dtype == np.uint8
            assert arrays['clipped_coordinates'].shape == (3,) and arrays['clipped_coordinates'].dtype == np.int64
    audit = json.loads((STATS/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['new_cases_checked'] == 100 and audit['bootstrap_scalar_values_checked'] == 30000
    d = json.loads((STATS/'statistics.json').read_text())
    assert len(d['rows']) == 100 and len(d['summaries']) == 2 and len(d['contrasts']) == 3
    cost = json.loads((STATS/'resource_ledger.json').read_text())
    plan = json.loads((ROOT/'layer_allocation_evidence/allocation.json').read_text())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = ROOT.parent/'FASHION_CNN_ALLOCATION_REPORT.md'
    assert not report.exists()
    lines = ['# Native CNN layer allocation at exactly matched sampling work', '',
             f'Recorded {now}. All 100 native cases, 4,718,250,000 counter entries, ten million final logits and all subsequent scalar/statistical checks are complete. The validation-selected schedule improves the fixed-cohort point accuracy from 78.5270% to 78.8002% at equal retained-sample node updates. The improvement is modest and does not close the gap to the 85.0860% FP32_F64 reference.', '',
             '| Condition | Accuracy (%) | Mean CE (nats) |', '|---|---:|---:|']
    for row in d['summaries']:
        lines.append(f"| {row['condition']} | {100*row['mean'][0]:.4f} | {row['mean'][1]:.8f} |")
    lines.append(f"| FP32_F64 | {100*d['baseline_mean'][0]:.4f} | {d['baseline_mean'][1]:.8f} |")
    lines += ['', '| Contrast | Accuracy difference (pp) [descriptive 95% interval] | CE difference (nats) [interval] |',
              '|---|---:|---:|']
    for row in d['contrasts']:
        point, ci = row['difference'], row['descriptive_95_interval']
        lines.append(f"| {row['contrast']} | {100*point[0]:+.4f} [{100*ci[0][0]:+.4f}, {100*ci[0][1]:+.4f}] | {point[1]:+.8f} [{ci[1][0]:+.8f}, {ci[1][1]:+.8f}] |")
    lines += ['', 'All five original frozen CNN model IDs, two sampling schedules, ten fresh inference seeds and all 10,000 historically consumed test images are retained. Five paired model units and independent within-condition seed resampling produce 5,000 hierarchical bootstrap draws. Images are fixed; intervals are descriptive, without multiplicity correction, fresh generalization confirmation or an equivalence test. No schedule is chosen or changed based on these test results.', '',
              'Selection uses only Gaussian validation sensitivity: one layer at T256-equivalent variance at a time while other layers use their nominal programmed means, over 5,000 training-validation images and ten seeds. The coefficient is max(mean isolated validation CE−nominal-mean CE,1e−4). Exhaustive optimization over all 20,009 feasible integer schedules minimizes the declared separable inverse-T proxy. Independent scalar enumeration verifies that proxy optimum. Its optimality is not a theorem about actual neural loss.', '',
              '| Model | Allocated conv1/conv2/head T | Summed layer rounds | Maximum T | Accuracy allocated−uniform (pp) |',
              '|---:|---|---:|---:|---:|']
    for mi, row in enumerate(plan['rows']):
        budgets = row['samples']
        delta = d['summaries'][1]['model_means'][mi][0]-d['summaries'][0]['model_means'][mi][0]
        lines.append(f"| {mi} | {' / '.join(map(str, budgets))} | {sum(budgets)} | {max(budgets)} | {100*delta:+.4f} |")
    lines += ['', 'Every schedule satisfies 72T_conv1+1152T_conv2+1440T_head=681,984 exactly, matching uniform (256,256,256) per eight-image bank. Both use 2,664 free weight nodes, identical finite eight-bit field commands, original per-output center/radius metadata, deterministic biases and ACT8 ranges, and integer output accumulation before external F64 readout/ReLU/maxpool. Uniform has 768 summed layer rounds and maximum T256. Different allocated sums/maxima are not measured chip latency.', '',
              f"Full production and public replay each execute {cost['full_production_retained_sample_updates']:,} retained-sample node updates plus {cost['full_warmup_updates_per_pass']:,} one-sweep warmup node updates. The required 20-case full-budget GPU preflight adds {cost['preflight_native_retained_updates']:,} native and the same public retained-sample updates, plus {cost['preflight_warmup_updates_per_pass']:,} warmup updates per pass. Earlier validation sensitivity and CPU-witness work remain separately retained.", '',
              'Nominal parameter payload is 3,392 bytes/model before schedule metadata. A uint16 uniform budget adds two bytes; three uint16 allocated budgets add six, giving 3,394 versus 3,398 logical bytes. Dynamic bits, 150,984 bytes of int32 bank output counters, activation/readout buffers and physical controllers are distinct. These quantities establish no physical memory/area/energy/latency advantage.', '',
              'Public replay regenerates every canonical counter/activation/mask/output receipt and every final logit; digests do not decode transient counters. The separate verifier recomputes 1,050,000 per-example scalar losses, all 100 metrics, all 30,000 bootstrap scalar values, summaries/intervals and resource identities. Source/input/completion/log hashes and all case schemas/coverage are reviewed separately here. Public replay is verification, not another independent accuracy replicate.', '',
              'The original frozen-model quality-gate miss remains unchanged, and this allocated frozen T256 result is still over six points below FP32. The separately fine-tuned CNN results include extra training and must not be conflated with this sampling-only comparison. A joint fine-tuning/allocation result has not been measured. The full research goal remains active for broader native model/device confirmation and physical costs.', '',
              '[Frozen native protocol](fashion_cnn_native/allocated_native_evidence/protocol.json), [all100cases](fashion_cnn_native/allocated_native_evidence/manifest.json), [public replay](fashion_cnn_native/allocated_native_evidence/native_audit.json), [statistics protocol](fashion_cnn_native/allocated_statistics_evidence/protocol.json), [all metrics and contrasts](fashion_cnn_native/allocated_statistics_evidence/statistics.json), [independent statistics audit](fashion_cnn_native/allocated_statistics_evidence/evidence_audit.json), [resource ledger](fashion_cnn_native/allocated_statistics_evidence/resource_ledger.json), [completion](fashion_cnn_native/allocated_statistics_evidence/completion.json).']
    report.write_text('\n'.join(lines)+'\n')
    write(STATS/'root_review.json', {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
                                   'all100case_hashes_schemas_and_full_coverage_checked': True,
                                   'all_source_input_completion_and_log_hashes_checked': True,
                                   'report_sha256': sha(report), 'scope': 'Post-closure artifact review, no new samples or physical measurements.'})
    paths = [report, STATS/'root_review.json', STATS/'completion.json', NATIVE/'pipeline.json', STATS/'pipeline.json']
    write(STATS/'review_completion.json', {'recorded_utc': now, 'status': 'full100case_native_sampling_allocation_reviewed_goal_active',
                                         'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All100native matched-budget outcomes reviewed; full report published', flush=True)


if __name__ == '__main__':
    main()
