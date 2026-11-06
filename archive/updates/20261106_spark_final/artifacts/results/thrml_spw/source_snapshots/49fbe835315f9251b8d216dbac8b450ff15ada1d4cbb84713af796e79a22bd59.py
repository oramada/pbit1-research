"""Full NumPy numerical and scalar loss audit, then separate allocation search."""
import datetime
import json
import math
import sys
import numpy as np
from results.thrml_spw.fashion_cnn_native.layer_allocation.common import (
    TRAIN, OUT, COUNTS, sha, write, guard, weights_for, numpy_forward)


def audit():
    assert not (OUT/'evidence_audit.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['complete'] and m['protocol_sha256'] == sha(OUT/'protocol.json')
    assert {(r['model'], r['layer'], r['seed']) for r in m['rows']} == {(mi, li, seed) for mi in range(5) for li in range(3) for seed in range(10)}
    assert len(m['rows']) == 150 and len(m['baseline_rows']) == 5
    data = np.load(TRAIN/'validation.npz')
    x, labels = data['inputs'].astype(float).reshape(625, 8, 1, 14, 14), data['labels']
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    checked = []
    for row in m['baseline_rows']+m['rows']:
        guard(p)
        assert sha(row['path']) == row['sha256']
        logits = np.load(row['path'])['logits']
        assert logits.shape == (5000, 10) and logits.dtype == np.float64
        model = np.load(train[row['model']]['path'])
        worst = 0.
        for first in range(0, 625, 16):
            n = min(16, 625-first)
            weights = weights_for(model, row['model'], row['seed'], row['layer'], first, n)
            output, _, _, _ = numpy_forward(x[first:first+n], model, weights)
            output = output.reshape(n*8, 10)
            saved = logits[first*8:(first+n)*8]
            np.testing.assert_allclose(output, saved, rtol=2e-12, atol=2e-11)
            np.testing.assert_array_equal(output.argmax(1), saved.argmax(1))
            worst = max(worst, float(np.max(abs(output-saved))))
        total = 0.
        for y, truth in zip(logits, labels):
            peak = max(y)
            total += peak+math.log(sum(math.exp(float(v-peak)) for v in y))-y[int(truth)]
        np.testing.assert_allclose(total/5000, row['mean_CE'], rtol=2e-13, atol=2e-14)
        checked.append({'name': row['name'], 'maximum_logit_error': worst})
        write(OUT/'audit_progress.json', {'rows': checked})
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'manifest_sha256': sha(OUT/'manifest.json'), 'rows': checked,
                                    'logit_values_checked': 7750000, 'scalar_losses_checked': 775000,
                                    'scope': 'All155validation cases including5nominal baselines. SeparateNumPy contractions/scalar losses; same specifiedGaussian draw helper. No test data, native execution or quality gain claim.'})


def verify_selection():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    m = json.loads((OUT/'manifest.json').read_text())
    selection = json.loads((OUT/'allocation.json').read_text())
    assert len(selection['rows']) == 5 and selection['manifest_sha256'] == sha(OUT/'manifest.json')
    for selected in selection['rows']:
        mi = selected['model']
        base = next(r['mean_CE'] for r in m['baseline_rows'] if r['model'] == mi)
        coefficients = [max(sum(r['mean_CE'] for r in m['rows'] if r['model'] == mi and r['layer'] == li)/10-base, 1e-4) for li in range(3)]
        np.testing.assert_allclose(coefficients, selected['coefficients'], rtol=2e-12, atol=1e-14)
        # Separate nested scalar enumeration, with different traversal/order.
        feasible, best = 0, None
        for third in range(64, 1025):
            for second in range(64, 1025):
                remaining = 2664*256-1152*second-1440*third
                if remaining % 72:
                    continue
                first = remaining//72
                if 64 <= first <= 1024:
                    feasible += 1
                    score = sum(coefficients[i]*256/[first, second, third][i] for i in range(3))
                    candidate = (score, (first, second, third))
                    if best is None or candidate < best:
                        best = candidate
        assert best[1] == tuple(selected['samples']) and feasible == selected['feasible_schedules']
        np.testing.assert_allclose(best[0], selected['surrogate'], rtol=2e-12, atol=1e-14)
        assert int(COUNTS@selected['samples']) == 2664*256
    guard(p)
    write(OUT/'selection_audit.json', {'passed': True, 'models': 5,
                                     'scope': 'Independent scalar exhaustive search verifies each floored validation coefficient, global discrete surrogate optimum, lexicographic tie-break and exact uniformT256 node-update equality. Surrogate optimality is not measured task improvement.'})
    paths = [OUT/n for n in ['protocol.json', 'manifest.json', 'evidence_audit.json', 'allocation.json', 'selection_audit.json']]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all5validation_allocations_numerically_and_optimally_checked_native_test_pending_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All5validation allocations independently checked; no native test scored', flush=True)


if __name__ == '__main__':
    {'audit': audit, 'selection': verify_selection}[sys.argv[1]]()
