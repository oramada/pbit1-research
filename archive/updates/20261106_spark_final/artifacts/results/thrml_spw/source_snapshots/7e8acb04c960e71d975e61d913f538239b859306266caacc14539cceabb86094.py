"""Independent model-unit accumulation check of figure values/intervals."""
import csv
import datetime
import json

import numpy as np

from results.thrml_spw.fashion_cnn_native.matched_figures.plot import BASE, WORD, OUT, sha, guard, write


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    data = json.loads((OUT/'data.json').read_text())
    saved = np.load(OUT/'figure_arrays.npz')
    base, word = np.load(BASE/'statistical_arrays.npz'), np.load(WORD/'statistical_arrays.npz')
    bm = base['bootstrap_models']
    for method, arrays, seed_key in [(0, base, 'bootstrap_seeds'), (1, word, 'bootstrap_word_seeds')]:
        for bi in range(3):
            distribution = np.zeros(5000)
            values = arrays['values'][bi, :, :, 0]
            for unit in range(5):
                distribution += values[bm[:, unit, None], arrays[seed_key][:, unit]].sum(1)*2
            np.testing.assert_allclose(distribution, saved['draws'][method, bi], rtol=2e-13, atol=2e-12)
            point = sum(sum(values[model]) for model in range(5))*2
            ci = np.quantile(distribution, [.025, .975])
            np.testing.assert_allclose(point, saved['points'][method, bi], rtol=2e-13, atol=2e-12)
            np.testing.assert_allclose(ci, saved['intervals'][method, bi], rtol=2e-13, atol=2e-12)
            row = data['rows'][method*3+bi]
            np.testing.assert_allclose([point, *ci], [row['accuracy_percent'], row['descriptive_low_percent'], row['descriptive_high_percent']], rtol=2e-13, atol=2e-12)
            assert row['scheduled_updates_per_weight_bank'] == row['pbits_per_weight']*row['samples_per_layer']
            assert row['scheduled_updates_full10000image_run'] == 2664*1250*row['scheduled_updates_per_weight_bank']
    with (OUT/'accuracy_resource_table.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    for row, expected in zip(rows, data['rows']):
        for key, value in expected.items():
            assert row[key] == str(value)
    statistics = json.loads((BASE/'statistics.json').read_text())
    for name, value in data['reference_mean_accuracy_percent'].items():
        assert value == 100*statistics['control_summary'][name][0]
    write(OUT/'evidence_audit.json', {'passed': True, 'source_sha256': sha(__file__),
                                   'bootstrap_accuracy_values_checked': 30000, 'table_rows_checked': 6,
                                   'native_sampler_calls': 0, 'fine_tuned_results_used': False,
                                   'scope': 'Separate model-unit summation checks every absolute bootstrap accuracy value, interval, point, CSV field and normalized-work identity. Rendered figure is a resource slice, not a physical Pareto proof.'})
    paths = [OUT/name for name in ['protocol.json', 'data.json', 'figure_arrays.npz', 'accuracy_resource_table.csv',
                                  'frozen_cnn_matched_work.svg', 'frozen_cnn_matched_work.png', 'evidence_audit.json']]
    guard(p)
    write(OUT/'completion.json', {'status': 'closed_CNN_resource_figure_numeric_audit_complete_goal_active',
                                 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All30000bootstrapaccuracyvalues and figure-table resources checked', flush=True)


if __name__ == '__main__':
    main()
