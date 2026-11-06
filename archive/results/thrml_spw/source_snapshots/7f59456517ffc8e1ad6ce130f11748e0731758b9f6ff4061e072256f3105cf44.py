"""Review full learned-command cohort only after its frozen pipeline succeeds."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, sha, write, guard

OUT = ROOT/'probability_cohort_evidence'


def main():
    assert not (OUT/'review_completion.json').exists()
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    assert pipeline['status'] == 'full_probability_finetuned_CNN_independently_audited_goal_active'
    completion = json.loads((OUT/'completion.json').read_text())
    for path, expected in completion['artifact_sha256'].items():
        assert sha(path) == expected, path
    protocol = json.loads((OUT/'protocol.json').read_text())
    guard(protocol)
    for stage in pipeline['stages']:
        assert stage['status'] == 'completed' and stage['returncode'] == 0
        assert sha(OUT/'logs'/(stage['stage']+'.log')) == stage['log_sha256']
    manifest = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    stats = json.loads((OUT/'statistics.json').read_text())
    checked = json.loads((OUT/'statistics_audit.json').read_text())
    assert manifest['complete'] and len(manifest['rows']) == 300
    assert {(r['method'], r['model'], r['T'], r['seed']) for r in manifest['rows']} == {(method, mi, t, seed) for method in range(2) for mi in range(5) for t in [64, 256, 1024] for seed in range(10)}
    assert audit['passed'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    assert len(audit['rows']) == 300 and audit['counter_entries_exact'] == 14154750000
    assert all(r['all17500_array_receipts_exact'] and r['all100000_final_logits_exact'] for r in audit['rows'])
    assert checked['passed'] and checked['case_metrics_checked'] == 300 and checked['bootstrap_values_checked'] == 330000
    assert checked['statistics_sha256'] == sha(OUT/'statistics.json')
    assert len(stats['contrasts']) == 33 and len(stats['summaries']) == 6
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            assert case['logits'].shape == (10000, 10) and case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            assert case['receipts'].shape == (1250, 14, 32) and case['receipts'].dtype == np.uint8
            assert case['clipped_coordinates'].shape == (3,) and case['clipped_coordinates'].dtype == np.int64
    # One warmup sweep precedes retained samples in SamplingSchedule(1,T,1).
    work = []
    for name, folder, cases, nodes in [('original_single', ROOT/'cohort_evidence', 150, 2664),
                                      ('four_node_word', ROOT/'word_comparison_evidence', 150, 10656),
                                      ('fine_tuned_single', OUT, 300, 2664)]:
        m = json.loads((folder/'manifest.json').read_text())
        assert m['complete'] and len(m['rows']) == cases
        retained = 447552000000 if cases == 300 else 223776000000
        warmup = cases*1250*nodes
        work.append({'cohort': name, 'cases': cases, 'physical_free_nodes_in_software_bank': nodes,
                     'retained_sample_scheduled_updates_per_pass': retained,
                     'warmup_scheduled_updates_per_pass': warmup,
                     'retained_plus_warmup_scheduled_updates_per_pass': retained+warmup})
    write(OUT/'warmup_accounting.json', {'rows': work,
                                      'scope': 'Full task passes only, production and independent public replay each incur the listed work. Prior preflights/training/reproduction/validation remain separately charged. Historical equal-update comparisons match retained-sample node updates; one warmup sweep adds different K-dependent work. This is software schedule accounting, not physical flips, controller programming, calibration, energy or latency.'})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = ROOT.parent/'FASHION_CNN_PROBABILITY_FINETUNE_REVIEW.md'
    assert not report.exists()
    lines = ['# Probability-finetuned native CNN: completed cohort review', '',
             f'Recorded {now}. All 300 cases have passed full public-state replay and independent controls/metrics/bootstrap verification. This separate review checks protocol inputs/sources, every case hash/schema, full coverage, completion artifacts and successful stage logs. All original frozen experiment files remain unchanged.', '',
             '| Training method | Samples per layer | Accuracy (%) | Mean CE (nats) |', '|---|---:|---:|---:|']
    for row in stats['summaries']:
        lines.append(f"| {row['method']} | {row['T']} | {100*row['mean'][0]:.4f} | {row['mean'][1]:.8f} |")
    lines += ['', 'Five starting CNNs, both predeclared training methods, all three budgets and ten inference seeds per model are retained. These are descriptive outcomes on historically consumed images; no favorable run, budget or method is substituted for the complete comparison. The original frozen-model quality-target miss remains unchanged.', '',
              'Expected-weight training optimizes the programmed stationary-mean network. Binomial-mean STE training adds a declared biased sampled-weight surrogate at T256. These are conventional CPU training procedures, followed by actual native inference. Both retain the original fixed per-output centers/radii and biases; the exported learned field commands are used directly. Learned means are not min/max refitted. Neither training method is claimed to be an unbiased native stochastic gradient estimator.', '',
              'Comparisons to the original frozen single-p-bit or four-node representations include additional training. They do not isolate representation effects under matched training. The four-node reference is internal and is not an assertion about Extropic hardware. The learned nominal-mean control is a plug-in deterministic CNN, not the expected output of the nonlinear stochastic network.', '',
              'All predeclared contrasts follow. Accuracy differences/intervals are percentage points; CE differences/intervals are nats. Each 95% interval uses 5,000 paired-model/within-method-seed hierarchical bootstrap draws with independent inference-seed resampling across methods/reference. Images stay fixed; no new generalization confirmation, equivalence test or multiplicity correction is claimed.', '',
              '| T | Contrast | Accuracy difference [interval] | CE difference [interval] |', '|---:|---|---:|---:|']
    for row in stats['contrasts']:
        d, ci = row['difference'], row['descriptive_95_interval']
        lines.append(f"| {row['T']} | {row['contrast']} | {100*d[0]:+.4f} [{100*ci[0][0]:+.4f}, {100*ci[0][1]:+.4f}] | {d[1]:+.8f} [{ci[1][0]:+.8f}, {ci[1][1]:+.8f}] |")
    lines += ['', 'The full fine-tuned cohort executes 447,552,000,000 retained-sample scheduled node updates in production and the same again in public replay. Its SamplingSchedule(1,T,1) adds 999,000,000 warmup node updates per full pass. The corresponding original single and four-node cohorts each have 223,776,000,000 retained-sample updates, with 499,500,000 and 1,998,000,000 additional warmup updates respectively. Historical matched-resource figures therefore match retained-sample work; total simulator sweep work also includes warmup. None of these counters measures physical flips or hardware energy/latency.', '',
              'One free dynamic p-bit per normalized weight still requires its finite control setting and metadata. The declared logical parameter payload remains 3,392 bytes/model, separate from dynamic states, 150,984 bytes of output counters per eight-image bank, activations/readout buffers, training latents and physical circuitry. No systems advantage follows solely from reduced dynamic node count. Physical cost and broader model/device confirmation remain open.', '',
              '[Frozen full report](FASHION_CNN_PROBABILITY_FINETUNE_REPORT.md), [all statistics](fashion_cnn_native/probability_cohort_evidence/statistics.json), [public replay](fashion_cnn_native/probability_cohort_evidence/native_audit.json), [independent statistics audit](fashion_cnn_native/probability_cohort_evidence/statistics_audit.json), [warmup accounting](fashion_cnn_native/probability_cohort_evidence/warmup_accounting.json), [original completion](fashion_cnn_native/probability_cohort_evidence/completion.json).', '',
              'The broader long-horizon goal remains active. A separate exact-budget native sample-allocation comparison is queued or running under its own frozen protocol.']
    report.write_text('\n'.join(lines)+'\n')
    write(OUT/'root_review.json', {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
                                 'all300case_hashes_and_schemas_checked': True, 'all_source_input_completion_and_log_hashes_checked': True,
                                 'completion_sha256': sha(OUT/'completion.json'), 'report_sha256': sha(report),
                                 'scope': 'Independent closure-artifact/source/coverage review after full frozen numerical/statistical audits. No new native draws, quality replicates or physical measurements.'})
    paths = [report, OUT/'completion.json', OUT/'root_review.json', OUT/'warmup_accounting.json', OUT/'pipeline.json']
    write(OUT/'review_completion.json', {'recorded_utc': now, 'status': 'full_learned_probability_CNN_reviewed_goal_active',
                                       'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All300learned-command CNN outcomes reviewed; companion report published', flush=True)


if __name__ == '__main__':
    main()
