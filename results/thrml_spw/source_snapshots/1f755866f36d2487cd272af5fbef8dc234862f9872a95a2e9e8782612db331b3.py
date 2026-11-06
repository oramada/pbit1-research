"""Review all completed fine-tuned native output files without task scoring."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT/'probability_cohort_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'production_root_review.json').exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    manifest = json.loads((OUT/'manifest.json').read_text())
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    assert manifest['complete'], 'Wait for the original producer to finish'
    for mapping in [protocol['source_sha256'], protocol['input_sha256']]:
        for file, expected in mapping.items():
            assert sha(file) == expected, file
    stage = next(r for r in pipeline['stages'] if r['stage'] == 'production')
    assert stage['status'] == 'completed' and stage['returncode'] == 0
    assert sha(OUT/'logs/production.log') == stage['log_sha256']
    assert manifest['protocol_sha256'] == sha(OUT/'protocol.json')
    assert manifest['parent_test_data_sha256'] == sha(ROOT/'cohort_evidence/test_data.npz')
    assert manifest['parent_completion_sha256'] == sha(ROOT/'probability_native_evidence/completion.json')
    expected = {(method, model, t, seed) for method in range(2) for model in range(5)
                for t in [64, 256, 1024] for seed in range(10)}
    assert len(manifest['rows']) == 300
    assert {(r['method'], r['model'], r['T'], r['seed']) for r in manifest['rows']} == expected
    assert len({r['path'] for r in manifest['rows']}) == 300
    assert sum(r['native_updates'] for r in manifest['rows']) == manifest['native_updates'] == 447552000000
    for row in manifest['rows']:
        assert row['native_updates'] == 1250*2664*row['T']
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            assert set(case.files) == {'receipts', 'logits', 'clipped_coordinates'}
            assert case['receipts'].shape == (1250, 14, 32) and case['receipts'].dtype == np.uint8
            assert case['logits'].shape == (10000, 10) and case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            assert case['clipped_coordinates'].shape == (3,) and case['clipped_coordinates'].dtype == np.int64
            assert np.all(case['clipped_coordinates'] >= 0)
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
    progress = OUT/'audit_progress.json'
    audit = json.loads(progress.read_text()) if progress.exists() else {'rows': [], 'counter_entries_exact': 0}
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'all300fine_native_output_files_checked_public_replay_pending',
              'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'protocol.json'),
              'manifest_sha256': sha(OUT/'manifest.json'), 'production_log_sha256': stage['log_sha256'],
              'all300_case_hashes_coverage_and_schemas_checked': True,
              'production_updates': manifest['native_updates'],
              'public_replay_cases_at_snapshot': len(audit['rows']),
              'public_replay_counter_entries_at_snapshot': audit['counter_entries_exact'],
              'task_metrics_computed_by_review': False,
              'scope': 'Saved output hashes, complete method/model/budget/seed coverage, schemas/finiteness/clipping metadata and successful production receipt. No labels/task metrics/new draws. Full independent public-state causal replay and control/statistical verification remain separate required gates; hashes do not decode transient counters.'}
    (OUT/'production_root_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print('All300fine native case files reviewed; full public replay/statistics remain pending')


if __name__ == '__main__':
    main()
