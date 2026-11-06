"""Review complete word production files without computing task metrics."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT/'word_comparison_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'production_root_review.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    m = json.loads((OUT/'manifest.json').read_text())
    pipe = json.loads((OUT/'pipeline.json').read_text())
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for name, expected in mapping.items():
            assert sha(name) == expected, name
    stage = next(row for row in pipe['stages'] if row['stage'] == 'production')
    assert stage['status'] == 'completed' and stage['returncode'] == 0
    assert sha(OUT/'logs/production.log') == stage['log_sha256']
    assert m['complete'] and m['native_updates'] == 223776000000
    assert m['protocol_sha256'] == sha(OUT/'protocol.json')
    assert m['parent_test_data_sha256'] == sha(ROOT/'cohort_evidence/test_data.npz')
    assert m['parent_completion_sha256'] == sha(ROOT/'cohort_evidence/completion.json')
    assert len(m['rows']) == 150
    expected = {(mi, t, seed) for mi in range(5) for t in [16, 64, 256] for seed in range(10)}
    assert {(r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    assert len({r['path'] for r in m['rows']}) == 150
    assert sum(r['native_updates'] for r in m['rows']) == m['native_updates']
    for row in m['rows']:
        assert row['matched_single_T'] == 4*row['T']
        assert row['native_updates'] == 1250*4*2664*row['T']
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            assert set(case.files) == {'receipts', 'logits', 'clipped_coordinates'}
            assert case['receipts'].shape == (1250, 14, 32) and case['receipts'].dtype == np.uint8
            assert case['logits'].shape == (10000, 10) and case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            assert case['clipped_coordinates'].shape == (3,) and case['clipped_coordinates'].dtype == np.int64
            assert np.all(case['clipped_coordinates'] >= 0)
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
    progress_path = OUT/'audit_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'rows': [], 'counter_entries_exact': 0}
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'status': 'all150_word_production_files_checked_public_audit_pending',
              'manifest_sha256': sha(OUT/'manifest.json'), 'protocol_sha256': sha(OUT/'protocol.json'),
              'production_log_sha256': stage['log_sha256'], 'all150_case_hashes_and_schemas_checked': True,
              'production_updates': m['native_updates'], 'public_audit_cases_at_snapshot': len(progress['rows']),
              'public_audit_counter_entries_at_snapshot': progress['counter_entries_exact'],
              'no_task_metrics_computed_in_this_review': True,
              'scope': 'Full saved-file hash/coverage/schema/finiteness/clipping-metadata review and successful terminal production receipt. Public raw-state causal replay and statistics are separate pending gates. Receipts do not decode transient counters; no accuracy result or systems claim.'}
    (OUT/'production_root_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print('All150word production case files reviewed; public replay/statistics remain pending')


if __name__ == '__main__':
    main()
