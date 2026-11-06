"""Closed-cohort integrity, coverage and retained-logit metric review."""
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
    assert not (OUT/'root_review.json').exists()
    completion = json.loads((OUT/'completion.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    for mapping in [completion['artifact_sha256'], protocol['source_sha256'], protocol['input_sha256']]:
        for file, expected in mapping.items():
            assert sha(file) == expected, file
    pipe = json.loads((OUT/'pipeline.json').read_text())
    assert pipe['status'] == 'full_matched_word_CNN_independently_audited_goal_active'
    for row in pipe['stages']:
        assert row['status'] == 'completed' and row['returncode'] == 0
        assert sha(OUT/'logs'/(row['stage']+'.log')) == row['log_sha256']
    manifest = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    statistics = json.loads((OUT/'statistics.json').read_text())
    stats_audit = json.loads((OUT/'statistics_audit.json').read_text())
    assert audit['passed'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    assert stats_audit['passed'] and stats_audit['statistics_sha256'] == sha(OUT/'statistics.json')
    assert len(manifest['rows']) == len(audit['rows']) == 150
    assert {(r['model'], r['T'], r['seed']) for r in manifest['rows']} == {(m, t, seed) for m in range(5) for t in [16, 64, 256] for seed in range(10)}
    assert all(r['all17500_bank_array_digests_exact'] for r in audit['rows'])
    assert sum(r['counter_entries_exact'] for r in audit['rows']) == audit['counter_entries_exact'] == 7077375000
    assert sum(r['public_updates'] for r in audit['rows']) == audit['public_updates'] == manifest['native_updates'] == 223776000000
    labels = np.load(ROOT/'cohort_evidence/test_data.npz')['labels']
    metrics = {row['name']: row for row in statistics['rows']}
    for row in manifest['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            z = case['logits']
            assert z.shape == (10000, 10) and np.isfinite(z).all()
            maximum = z.max(1)
            loss = np.log(np.exp(z-maximum[:, None]).sum(1))+maximum-z[np.arange(10000), labels]
            v = [np.mean(z.argmax(1)==labels), loss.mean()]
            np.testing.assert_allclose(v, [metrics[row['name']]['accuracy'], metrics[row['name']]['mean_CE']], rtol=2e-12, atol=2e-13)
    result = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'completion_sha256': sha(OUT/'completion.json'),
              'case_metrics_recomputed': 150, 'final_logit_values_checked': 15000000,
              'public_replay_counter_entries_exact': 7077375000,
              'producer_updates': 223776000000, 'public_updates': 223776000000,
              'independent_statistics_audit_bootstrap_values': stats_audit['bootstrap_values_checked'],
              'word_minus_single': [c for c in statistics['contrasts'] if c['contrast'] == 'word_minus_single'],
              'scope': 'Completion/source/input/case/log integrity and coverage review, with independent retained-logit metric recomputation. Full public raw replay and all120000bootstrapvalues are checked by separate completed audits. Largest-budget accuracy/CE intervals include zero, not formal equivalence or a physical systems win.'}
    (OUT/'root_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Closed150case matched-word CNN integrity and metrics reviewed')


if __name__ == '__main__':
    main()
