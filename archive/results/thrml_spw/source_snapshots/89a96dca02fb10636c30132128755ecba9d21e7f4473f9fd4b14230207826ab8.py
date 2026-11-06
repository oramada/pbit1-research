"""Synthetic-only check of frozen RGB statistics; no native or task evidence."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.cohort import statistics as target

DEST = Path('results/thrml_spw/cifar_cnn_native/statistics_fixture_evidence')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def independent_metric(logits, labels):
    accuracy = sum(int(max(range(10), key=lambda j: float(row[j])) == int(label))
                   for row, label in zip(logits, labels))/len(labels)
    losses = []
    for row, label in zip(logits, labels):
        maximum = max(float(v) for v in row)
        losses.append(maximum+math.log(sum(math.exp(float(v)-maximum) for v in row))-float(row[label]))
    return np.array([accuracy, math.fsum(losses)/len(labels)])


def main():
    assert not DEST.exists(), 'Synthetic fixture is one-shot; preserve existing evidence.'
    DEST.mkdir()
    (DEST/'README.md').write_text('# SYNTHETIC STATISTICS FIXTURE ONLY\n\nNo neural model, dataset evaluation, p-bit draw or native replay is performed. The audit success flags and scheduled counts are explicit fixture placeholders needed to exercise the frozen statistical entry point. They are not scientific execution receipts. Output files stay in this separate namespace.\n')
    original = target.OUT
    protocol = json.loads((original/'protocol.json').read_text())
    protocol['fixture_only'] = True
    protocol['statistics'] = 'SYNTHETIC FIXTURE ONLY: asymmetric generated logits, no task quality or native evidence.'
    protocol['source_sha256'][str(Path(__file__))] = sha(__file__)
    write(DEST/'protocol.json', protocol)
    rng = np.random.default_rng(1757520)
    labels = np.tile(np.arange(10), 2)
    np.savez_compressed(DEST/'test_data.npz', labels=labels)
    base = np.empty((10, 5, 2))
    values = np.empty((3, 5, 10, 2))
    controls = {}
    for ci in range(10):
        for mi in range(5):
            z = rng.normal(size=(20, 10))*(.7+.1*mi)
            z[np.arange(20), labels] += .05*ci-.07*mi
            controls[f'c{ci}_model{mi}'] = z
            base[ci, mi] = independent_metric(z, labels)
    np.savez_compressed(DEST/'control_logits.npz', **controls)
    write(DEST/'controls.json', {'fixture_only': True})
    rows = []
    for mi in range(5):
        for bi, t in enumerate(protocol['samples']):
            for seed in range(10):
                z = rng.normal(size=(20, 10))*(1+.03*seed)
                z[np.arange(20), labels] += .2*bi+.04*mi-.02*seed
                values[bi, mi, seed] = independent_metric(z, labels)
                name = f'model{mi}_T{t}_seed{seed}'
                path = DEST/(name+'.npz')
                np.savez_compressed(path, logits=z)
                rows.append({'model': mi, 'T': t, 'seed': seed, 'name': name,
                             'path': str(path), 'sha256': sha(path),
                             'clipped_coordinates': [0, 0, 0],
                             'all_receipts_and_final_logits_exact': True})
    write(DEST/'manifest.json', {'fixture_only': True, 'complete': True, 'rows': rows})
    write(DEST/'native_audit.json', {'fixture_only': True, 'passed': True,
          'complete': True, 'rows': rows, 'manifest_sha256': sha(DEST/'manifest.json'),
          'scheduled_updates': 329952000000, 'counter_entries': 9237375000})
    # Redirect only output paths. Execute the unchanged source and real hash guard.
    target.OUT = DEST
    try:
        target.main()
    finally:
        target.OUT = original
    d = json.loads((DEST/'statistics.json').read_text())
    with np.load(DEST/'statistical_arrays.npz') as arrays:
        np.testing.assert_allclose(arrays['values'], values, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(arrays['baseline_metrics'], base, rtol=2e-13, atol=2e-14)
        # Reproduce index draws, then enumerate every scalar of every contrast.
        index_rng = np.random.default_rng(protocol['bootstrap_seed'])
        bm = index_rng.integers(0, 5, (5000, 5))
        bs = index_rng.integers(0, 10, (5000, 5, 10))
        np.testing.assert_array_equal(bm, arrays['bootstrap_models'])
        np.testing.assert_array_equal(bs, arrays['bootstrap_seeds'])
        seen = set()
        for c in d['contrasts']:
            bi, ci = c['budget_index'], c['control_index']
            assert (bi, ci) not in seen
            seen.add((bi, ci))
            assert c['control'] == target.CONTROLS[ci]
            assert c['T'] == protocol['samples'][bi]
            reference = np.empty((5000, 2))
            for draw in range(5000):
                for metric in range(2):
                    reference[draw, metric] = math.fsum(
                        float(values[bi, int(bm[draw, unit]), int(bs[draw, unit, sample]), metric]
                              -base[ci, int(bm[draw, unit]), metric])
                        for unit in range(5) for sample in range(10))/50
            np.testing.assert_allclose(reference, arrays[f'budget{bi}_control{ci}'], rtol=2e-11, atol=2e-13)
            np.testing.assert_allclose(c['difference'], (values[bi]-base[ci, :, None]).mean((0, 1)), rtol=2e-11, atol=2e-13)
            np.testing.assert_allclose(c['descriptive_95_interval'], np.quantile(reference, [.025, .975], axis=0).T, rtol=2e-11, atol=2e-13)
        assert seen == {(bi, ci) for bi in range(3) for ci in range(10)}
    write(DEST/'fixture_completion.json', {'fixture_only': True, 'passed': True,
          'source_sha256': sha(__file__), 'frozen_statistics_source_sha256': sha(target.__file__),
          'independent_case_metrics': 150, 'independent_control_metrics': 50,
          'bootstrap_scalar_values_enumerated': 300000, 'all_contrasts': 30,
          'native_calls': 0, 'task_quality_claim': False,
          'artifact_sha256': {str(p): sha(p) for p in DEST.iterdir() if p.is_file()}})
    print('SYNTHETIC ONLY: all30contrasts/300000bootstrap scalar values match independent enumeration.')


if __name__ == '__main__':
    main()
