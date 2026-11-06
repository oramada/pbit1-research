"""Review full produced T4096 coverage/storage before public replay acceptance."""
import datetime
import json
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_high_budget.common import (
    ROOT, OUT, PARENT, sha, write, guard, parent_closed)
from results.thrml_spw.cifar_cnn_native.probability_high_budget.pipeline import gpu_jobs


def main():
    assert not (OUT/'production_review.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    state = json.loads((OUT/'pipeline.json').read_text())
    production = next(s for s in state['stages'] if s['stage'] == 'production')
    assert production['status'] == 'completed' and production['returncode'] == 0
    assert sha(OUT/'logs/production.log') == production['log_sha256']
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['complete'] and m['protocol_sha256'] == sha(OUT/'protocol.json')
    assert sha(OUT/'mean_logits.npz') == sha(PARENT/'mean_logits.npz')
    expected = [(method, model, 4096, seed) for method in range(2) for model in range(5) for seed in range(10)]
    assert [(r['method'], r['model'], r['T'], r['seed']) for r in m['rows']] == expected
    assert m['native_updates'] == sum(r['native_updates'] for r in m['rows']) == 2011136000000
    stored_raw_bytes = 0
    clipped = np.zeros((2, 3), np.int64)
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        assert row['native_updates'] == 1250*3928*4096
        with np.load(row['path']) as case:
            assert set(case.files) == {'receipts', 'logits', 'clipped_coordinates'}
            assert case['receipts'].shape == (1250, 14, 32) and case['receipts'].dtype == np.uint8
            assert case['logits'].shape == (10000, 10) and case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            assert case['clipped_coordinates'].shape == (3,) and case['clipped_coordinates'].dtype == np.int64
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
            clipped[row['method']] += case['clipped_coordinates']
            stored_raw_bytes += sum(case[name].nbytes for name in case.files)
    assert stored_raw_bytes == 136002400
    jobs = gpu_jobs()
    assert len(jobs) == 1 and jobs[0]['module'].endswith('.probability_high_budget.run')
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'manifest_sha256': sha(OUT/'manifest.json'),
        'protocol_sha256': sha(OUT/'protocol.json'), 'cases_checked': 100,
        'retained_scheduled_updates': 2011136000000, 'warmup_scheduled_updates': 491000000,
        'counter_entries_scheduled_for_full_replay': 6158250000,
        'all_case_array_bytes_including_clipped_counts': stored_raw_bytes,
        'case_receipt_and_logit_subtotal_bytes': 136000000, 'clipped_count_bytes': 2400,
        'storage_note': 'Protocol136000000raw-case subtotal counts receipts/logits only; allarrays including clipped coordinates are136002400B. Separate137MBcase-with-overhead bound/250MBtotal allowance covers this. Frozen protocol retained unchanged.',
        'clipped_coordinates_by_method': clipped.tolist(), 'production_stage': production,
        'actual_GPU_jobs_observed': jobs,
        'scope': 'Full produced coverage/hash/schema/finite-logit/resource review without labels/task metrics. Raw counter digest receipts have not yet passed full public replay; no task quality/BF16 equivalence or physical systems claim.'}
    write(OUT/'production_review.json', review)
    report = ROOT.parent/'CIFAR_CNN_HIGH_BUDGET_PRODUCTION_REVIEW.md'
    assert not report.exists()
    report.write_text('''# Full T4096 RGB inference production complete

All100planned cases were produced, with exact two-method/five-model/ten-MC-seed coverage. The original production stage completed with exit0 and its log hash matches. Every case file hash, receipt schema, finite10000x10F64logits and clipped-coordinate record was checked without loading labels or calculating task accuracy. Reused nominal-mean controls remain byte-identical to the independently audited parent.

Full native inference used2011136000000retained scheduled node updates plus491000000warmup updates. Public replay must independently regenerate6158250000integer counter entries and10millionfinal logits before statistics/acceptance. Production correctness is not BF16-equivalent task quality.

Stored uncompressed arrays total136002400bytes:136000000receipt/logit bytes plus2400clipped-count bytes. The frozen protocol's136000000raw-array subtotal omitted the small clipped-count arrays; its separate137MBcase-with-overhead and250MBtotal output bounds cover them. The discrepancy is recorded without changing the frozen protocol. Evidence storage is not physical model memory or a systems advantage.

The same launched pipeline has moved into independent public-state replay. Complete replay and independent metrics/BF16/bootstrap checks remain required. No task accuracy has been accepted or released from this study.

[Production review](cifar_cnn_native/probability_high_budget_evidence/production_review.json) · [Pipeline](cifar_cnn_native/probability_high_budget_evidence/pipeline.json) · [Fixed protocol](CIFAR_CNN_HIGH_BUDGET_PROTOCOL_REPORT.md).

The broader research goal remains active.
''')
    write(OUT/'production_review_completion.json', {'passed': True, 'artifact_sha256': {
        str(OUT/'production_review.json'): sha(OUT/'production_review.json'), str(report): sha(report)}})
    print('All100producedT4096case hashes/schema/coverage checked; full public replay pending')


if __name__ == '__main__':
    main()
