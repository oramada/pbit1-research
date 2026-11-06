"""Seal a root review of the unchanged same-fit BF16 control experiment."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import sha, seal, guard


def main():
    root = Path('results/thrml_spw/cifar_cnn_native')
    out = root/'same_fit_bf16_evidence'
    assert not (out/'review_completion.json').exists()
    assert (root.parent/'source_snapshots'/(sha(__file__)+'.py')).read_bytes() == Path(__file__).read_bytes()
    seal(out/'completion.json')
    p = json.loads((out/'protocol.json').read_text()); guard(p)
    audit = json.loads((out/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['control_paths'] == 30 and audit['logits_checked'] == 3000000
    assert audit['contrast_rows'] == 24 and audit['bootstrap_scalar_values_checked'] == 240000
    assert audit['new_native_updates'] == 0 and audit['statistics_sha256'] == sha(out/'statistics.json')
    assert len(audit['control_max_errors']) == 30
    s = json.loads((out/'statistics.json').read_text())
    a = np.load(out/'statistical_arrays.npz'); metrics = a['control_metrics']
    assert metrics.shape == (2, 3, 5, 2)
    with np.load(out/'logits.npz') as logits:
        assert set(logits.files) == {f'method{m}_c{ci}_model{mi}'
            for m in range(2) for ci in range(3) for mi in range(5)}
    high = np.load(root/'probability_high_budget_evidence/statistical_arrays.npz')
    low = np.load(root/'probability_cohort_evidence/statistical_arrays.npz')
    assert len(s['contrasts']) == 24
    seen = set()
    for c in s['contrasts']:
        m, ci, t = c['method_index'], c['control_index'], c['T']
        assert (m, ci, t) not in seen
        seen.add((m, ci, t))
        assert c['method'] == p['methods'][m] and c['control'] == p['controls'][ci]
        values = high['values'][m] if t == 4096 else low['values'][m, p['samples'].index(t)]
        np.testing.assert_allclose(values.mean((0, 1))-metrics[m, ci].mean(0), c['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(a[c['array_key']], [.025, .975], axis=0).T,
            c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert seen == {(m, ci, t) for m in range(2) for ci in range(3) for t in p['samples']}
    paths = [out/'completion.json', out/'protocol.json', out/'statistics.json', out/'evidence_audit.json',
             root.parent/'CIFAR_CNN_SAME_FIT_BF16_REPORT.md']
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'contrasts_unit_checked': 24,
        'bootstrap_scalar_values_interval_checked': 240000, 'new_native_updates': 0,
        'artifact_sha256': {str(path): sha(path) for path in paths},
        'scope': 'Root completion/source/input/coverage/unit/interval review after independent full graph/metric/bootstrap audit. No new stochastic evidence or formal equivalence/physical claim.'}
    (out/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (out/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(out/'root_review.json'): sha(out/'root_review.json')}}, indent=2)+'\n')
    print('Same-fit BF16 closure reviewed: 30 controls, 24 comparisons, zero new stochastic draws')


if __name__ == '__main__':
    main()
