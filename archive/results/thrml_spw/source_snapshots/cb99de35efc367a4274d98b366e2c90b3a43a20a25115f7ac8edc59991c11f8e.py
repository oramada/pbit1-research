"""Read-only closure review of completed CNN cohorts; no sampler calls."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    out = ROOT / 'cohort_evidence'
    target = out / 'root_review.json'
    assert not target.exists(), 'Preserve an existing closure review'
    checks = {}
    for folder in ['cohort_evidence', 'precision_evidence']:
        base = ROOT / folder
        closed = read(base / 'completion.json')
        protocol = read(base / 'protocol.json')
        for mapping in [closed['artifact_sha256'], protocol['source_sha256'], protocol['input_sha256']]:
            for path, expected in mapping.items():
                actual = sha(path)
                assert actual == expected, (folder, path, expected, actual)
                checks[path] = actual
        pipe = read(base / 'pipeline.json')
        assert all(row['returncode'] == 0 and row['status'] == 'completed' for row in pipe['stages'])
        for row in pipe['stages']:
            if 'log_sha256' in row:
                assert sha(base / 'logs' / (row['stage'] + '.log')) == row['log_sha256']
    manifest = read(out / 'manifest.json')
    audit = read(out / 'native_audit.json')
    statistics = read(out / 'statistics.json')
    protocol = read(out / 'protocol.json')
    assert audit['passed'] and read(out / 'statistics_audit.json')['passed']
    assert audit['producer_manifest_sha256'] == sha(out / 'manifest.json')
    expected = {(model, t, seed) for model in range(5) for t in [64, 256, 1024] for seed in range(10)}
    assert len(manifest['rows']) == 150
    assert {(r['model'], r['T'], r['seed']) for r in manifest['rows']} == expected
    assert len(audit['rows']) == 150
    assert all(r['all17500_array_receipts_exact'] and r['all100000_final_logits_exact'] for r in audit['rows'])
    assert sum(r['counter_entries_exact'] for r in audit['rows']) == 7077375000
    assert sum(r['public_updates'] for r in audit['rows']) == 223776000000
    metrics = {row['name']: row for row in statistics['rows']}
    labels = np.load(out / 'test_data.npz')['labels']
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            logits = case['logits']
            assert logits.shape == (10000, 10) and np.isfinite(logits).all()
            assert case['receipts'].shape == (1250, 14, 32)
            shifted = logits - logits.max(1)[:, None]
            ce = np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(10000), labels])
            accuracy = np.mean(logits.argmax(1) == labels)
            saved = metrics[row['name']]
            np.testing.assert_allclose([accuracy, ce], [saved['accuracy'], saved['mean_CE']], rtol=2e-12, atol=2e-13)
    selected = [row for row in statistics['contrasts'] if row['T'] == 1024 and row['control'] in ['FP32_F64', 'ACT8_INT4_minmax']]
    result = {
        'passed': True,
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sha(__file__),
        'closure_source_input_hashes_checked': checks,
        'cases_checked': 150,
        'final_logits_checked': 15000000,
        'native_replay_counter_entries_exact': audit['counter_entries_exact'],
        'native_updates': protocol['native_updates'],
        'public_replay_updates': audit['public_updates'],
        'summaries': statistics['summaries'],
        'largest_budget_selected_contrasts': selected,
        'scope': 'Closure/source/case-hash review and independent retained-logit metric recomputation. Full raw-state recovery and all bootstrap/control checks are separately completed by the sealed native/statistical audits. No new simulator draws or fresh-data confirmation. The approximate 1–2 percentage-point CNN quality target is not declared passed by the 2.0222-point largest-budget observed loss. Matched four-node CNN production and physical-system advantage remain open.'
    }
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('Reviewed both completed closures, all source/input hashes and all150 CNN case metrics')


if __name__ == '__main__':
    main()
