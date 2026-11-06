"""Metadata/file integrity review after all fixed allocated-native cases exist."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, sha, write, guard

OUT = ROOT/'allocated_native_evidence'


def main():
    assert not (OUT/'production_review_completion.json').exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    guard(protocol)
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    produced = next(r for r in pipeline['stages'] if r['stage'] == 'run')
    assert produced['status'] == 'completed' and produced['returncode'] == 0
    assert produced['log_sha256'] == sha(OUT/'logs/run.log')
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['complete'] and manifest['protocol_sha256'] == sha(OUT/'protocol.json')
    expected = {(ci, mi, seed) for ci in range(2) for mi in range(5) for seed in range(10)}
    assert len(manifest['rows']) == 100 and {(r['condition'], r['model'], r['seed']) for r in manifest['rows']} == expected
    clipped = np.zeros((2, 3), np.int64)
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        target = [256]*3 if row['condition'] == 0 else protocol['schedules'][row['model']]
        assert row['budgets'] == target and sum(n*t for n, t in zip([72, 1152, 1440], target)) == 681984
        with np.load(row['path']) as case:
            assert set(case.files) == {'logits', 'receipts', 'clipped_coordinates'}
            assert case['logits'].shape == (10000, 10) and case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            assert case['receipts'].shape == (1250, 14, 32) and case['receipts'].dtype == np.uint8
            assert case['clipped_coordinates'].shape == (3,) and case['clipped_coordinates'].dtype == np.int64
            assert np.all(case['clipped_coordinates'] >= 0)
            clipped[row['condition']] += case['clipped_coordinates']
    assert manifest['scheduled_updates'] == 85248000000 and manifest['counter_entries'] == 4718250000
    guard(protocol)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    review = {'passed': True, 'recorded_utc': now, 'source_sha256': sha(__file__),
              'manifest_sha256': sha(OUT/'manifest.json'), 'producer_log_sha256': produced['log_sha256'],
              'all100case_hashes_schemas_finiteness_coverage_budgets_checked': True,
              'clipped_coordinates_by_condition': clipped.tolist(),
              'scope': 'Production integrity only: no labels, accuracy, loss, new samples or native/public correctness acceptance. Full public replay and separate frozen statistics remain required.'}
    write(OUT/'production_root_review.json', review)
    report = ROOT.parent/'FASHION_CNN_ALLOCATION_PRODUCTION_REVIEW.md'
    assert not report.exists()
    report.write_text('''# Exact-budget native CNN: all production cases complete

All 100 predeclared cases completed with producer exit code zero: five original frozen CNNs, two schedules and ten inference seeds, each evaluated on all 10,000 historically consumed images. Every saved case hash/schema, finite logit array, clipping record, condition/model/seed identity and per-model budget has passed this production integrity review. No task labels or accuracy/loss metrics are used by this review.

Both schedules use exactly 681,984 retained-sample weight-node updates per eight-image bank. Full production totals 85,248,000,000 retained-sample scheduled updates and 4,718,250,000 transient counter entries. The one-sweep warmup adds 333,000,000 node updates, separately charged. These are software schedule counters, not measured physical flips, energy or latency.

The three-layer schedules were fixed from validation-only Gaussian sensitivity proxies and independently verified before this test run. All five schedules and both conditions remain retained. No schedule is changed or selected based on test outcomes.

Independent public-state replay is a separate required stage. It must regenerate every canonical counter/activation/mask/output receipt and every final logit. The already-frozen statistical checker then computes all 100 metrics and all three contrasts and verifies scalar losses/bootstrap/resource identities. Production completion alone establishes no accuracy improvement or new systems benefit.

[Frozen protocol](fashion_cnn_native/allocated_native_evidence/protocol.json), [all100case files](fashion_cnn_native/allocated_native_evidence/manifest.json), [production integrity review](fashion_cnn_native/allocated_native_evidence/production_root_review.json), [live stages](fashion_cnn_native/allocated_native_evidence/pipeline.json), [prospective statistics](FASHION_CNN_ALLOCATION_STATISTICS_PROTOCOL_REPORT.md).

The broader research goal remains active. Full public replay, task statistics and their independent checks remain pending.
''')
    paths = [report, OUT/'production_root_review.json', OUT/'manifest.json', OUT/'logs/run.log']
    write(OUT/'production_review_completion.json', {'recorded_utc': now, 'status': 'all100production_cases_integrity_checked_public_statistics_pending_goal_active',
                                                   'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All100production case integrity checks passed; public/statistics pending', flush=True)


if __name__ == '__main__':
    main()
