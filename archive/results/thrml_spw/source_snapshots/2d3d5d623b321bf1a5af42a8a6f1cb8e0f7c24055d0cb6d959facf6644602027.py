"""Independent scalar/unit/coverage checks for the frozen RGB figure data."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'frozen_figure_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, expected in p[group].items():
            assert sha(path) == expected, path
    data = json.loads((OUT/'figure_data.json').read_text())
    assert len(data['rows']) == 17 and data['bootstrap_values_sha256'] == sha(OUT/'bootstrap_values.npz')
    base = np.load(ROOT/'cohort_evidence/statistical_arrays.npz')
    precision = np.load(ROOT/'precision_evidence/statistical_arrays.npz')
    saved = np.load(OUT/'bootstrap_values.npz')
    csv_rows = list(csv.DictReader((OUT/'sampling_curve.csv').open()))
    assert len(csv_rows) == 17
    bm, bs = base['bootstrap_models'], base['bootstrap_seeds']
    expected_names = ['Single_pbit_T64', 'Single_pbit_T256', 'Single_pbit_T1024',
        'FP32_F64', 'FP32_F32', 'ACT8_FP32', 'ACT8_INT2_minmax', 'ACT8_INT4_minmax', 'ACT8_INT8_minmax',
        'ACT8_INT2_symmetric', 'ACT8_INT4_symmetric', 'ACT8_INT8_symmetric', 'ACT8_nominal_programmed_mean',
        'FP16_weights_F64', 'BF16_weights_F64', 'FP16_Torch_CPU', 'BF16_Torch_CPU']
    assert [r['condition'] for r in data['rows']] == expected_names
    for ri, row in enumerate(data['rows']):
        source = base['values'][ri] if ri < 3 else base['baseline_metrics'][ri-3] if ri < 13 else precision['control_metrics'][ri-13]
        independent = np.empty((5000, 2))
        for draw in range(5000):
            for metric in range(2):
                independent[draw, metric] = math.fsum(float(source[bm[draw, unit], bs[draw, unit, sample], metric]) for unit in range(5) for sample in range(10))/50 if ri < 3 else math.fsum(float(source[bm[draw, unit], metric]) for unit in range(5))/5
        np.testing.assert_allclose(independent, saved[f'row{ri}'], rtol=2e-12, atol=2e-13)
        mean = source.mean((0, 1)) if ri < 3 else source.mean(0)
        interval = np.quantile(independent, [.025, .975], axis=0).T
        np.testing.assert_allclose(mean, row['mean'], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(interval, row['interval'], rtol=2e-12, atol=2e-13)
        c = csv_rows[ri]
        assert c['condition'] == row['condition'] and c['kind'] == row['kind']
        assert c['samples_per_layer'] == str(row['T']) if ri < 3 else c['samples_per_layer'] == 'NA'
        for name, value in [('accuracy_fraction', mean[0]), ('accuracy_lo', interval[0, 0]), ('accuracy_hi', interval[0, 1]),
                            ('CE_nats', mean[1]), ('CE_lo', interval[1, 0]), ('CE_hi', interval[1, 1])]:
            np.testing.assert_allclose(float(c[name]), value, rtol=2e-12, atol=2e-13)
        assert int(c['retained_weight_node_updates_per8imagebank']) == (3928*row['T'] if ri < 3 else 0)
    report = ROOT.parent/'CIFAR_CNN_FROZEN_SAMPLING_FIGURE.md'
    assert not report.exists()
    report.write_text('''# Verified frozen RGB sampling curve

![Frozen compact RGB CIFAR sampling curve](cifar_cnn_native/frozen_figure_evidence/frozen_cifar_sampling.png)

All three predetermined native budgets and14conventional controls are retained in data/CSV. The plot shows native accuracy/loss and preselected FP32/ACT8min-maxINT4/INT8reference levels. Independent scalar enumeration verifies all170000bootstrap values and every point/interval/CSVunit. Plotted accuracy is percent; stored accuracy is a fraction. Absolute model/seed intervals do not replace paired contrasts or establish fresh-data confirmation/equivalence.

AtT1024 accuracy53.5644%versus56.2580%FP32 leaves2.6936points loss; the approximate1–2point target is missed. The benchmark is a compact16x16RGB model with a moderate baseline, not competitive modern CIFARquality. Both accuracy and loss and all lower-budget outcomes remain visible.

Samples/retained-node updates are software counts. Conventional reference lines do not imply zero arithmetic cost; no physical energy, latency, area, controller cost or systems improvement is demonstrated. The four-node and trained-model cohorts require their own full audits and are excluded from this frozen-only figure.

[SVG](cifar_cnn_native/frozen_figure_evidence/frozen_cifar_sampling.svg) · [PNG](cifar_cnn_native/frozen_figure_evidence/frozen_cifar_sampling.png) · [All17rows CSV](cifar_cnn_native/frozen_figure_evidence/sampling_curve.csv) · [All paired CIFAR contrasts](CIFAR_CNN_NATIVE_REVIEW.md)
''')
    audit = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_sha256': sha(__file__), 'conditions_checked': 17, 'bootstrap_scalar_values_checked': 170000,
             'scope': 'Independent derivative data/CSV/bootstrap/unit checks only; not new neural cases or physical measurements.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    paths = [OUT/name for name in ['protocol.json', 'figure_data.json', 'bootstrap_values.npz', 'sampling_curve.csv',
             'frozen_cifar_sampling.svg', 'frozen_cifar_sampling.png', 'evidence_audit.json']]+[report]
    (OUT/'completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {str(path): sha(path) for path in paths}}, indent=2)+'\n')
    print('All17conditions/170000bootstrap scalars/CSVunits independently checked', flush=True)


if __name__ == '__main__':
    main()
