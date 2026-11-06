"""Read-only review of all learned-command native validation receipts."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path('results/thrml_spw/fashion_cnn_native/probability_native_evidence')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'root_review.json').exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    completion = json.loads((OUT/'completion.json').read_text())
    for mapping in [completion['artifact_sha256'], protocol['source_sha256'], protocol['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    assert pipeline['status'] == 'learned_probability_native_validation_audited_goal_active'
    assert len(pipeline['stages']) == 1
    row = pipeline['stages'][0]
    assert row['status'] == 'completed' and row['returncode'] == 0
    assert row['log_sha256'] == sha(OUT/'execution.log')
    audit = json.loads((OUT/'evidence_audit.json').read_text())
    assert audit['passed'] and len(audit['rows']) == 30
    expected = {(m, method, t) for m in range(5) for method in protocol['methods'] for t in [64, 256, 1024]}
    assert {(r['model'], r['method'], r['T']) for r in audit['rows']} == expected
    assert all(r['all28_bank_arrays_exact'] for r in audit['rows'])
    assert sum(r['counter_entries_exact'] for r in audit['rows']) == audit['counter_entries_exact'] == 2264760
    assert sum(r['native_updates'] for r in audit['rows']) == sum(r['public_updates'] for r in audit['rows']) == 71608320
    for row in audit['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            names = {f'{prefix}_{li}' for li in range(3) for prefix in ['gate', 'column', 'codes', 'output']} | {'mask_0', 'mask_1'}
            assert set(case.files) == names
            assert case['output_2'].shape == (2, 8, 10)
            assert sum(case[f'gate_{li}'].size+case[f'column_{li}'].size for li in range(3)) == 75492
            for name in names:
                value = case[name]
                assert np.isfinite(value).all()
                if name.startswith(('gate_', 'column_')):
                    assert value.dtype == np.int32 and value.min() >= 0
                elif name.startswith(('codes_', 'mask_')):
                    assert value.dtype == np.uint8
                else:
                    assert value.dtype == np.float64
    result = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'completion_sha256': sha(OUT/'completion.json'),
              'validation_cases_checked': 30, 'counter_entries_exact_in_public_replay': 2264760,
              'native_updates': 71608320, 'public_updates': 71608320, 'new_sampler_invocations_by_review': 0,
              'scope': 'Complete closure/source/input/log/case-hash/coverage/array-schema review. Separate completed native/public code paths establish exact raw-state recovery; this review adds no draws/labels/task metrics/physical evidence. Full task cohort remains separate.'}
    (OUT/'root_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print('All30learned-command native validation cases and closure reviewed')


if __name__ == '__main__':
    main()
