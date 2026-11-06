"""Independent unit accumulation and export/resource checks for every plot value."""
import csv
import datetime
import json
import numpy as np
from results.thrml_spw.fashion_cnn_native.finetune_figures.common import BASE, WORD, FINE, OUT, LABELS, sha, write, guard


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    d = json.loads((OUT/'data.json').read_text())
    assert d['protocol_sha256'] == sha(OUT/'protocol.json') and d['figure_arrays_sha256'] == sha(OUT/'figure_arrays.npz')
    a = np.load(OUT/'figure_arrays.npz')
    base, fine, word = [np.load(folder/'statistical_arrays.npz') for folder in [BASE, FINE, WORD]]
    word_cost = json.loads((WORD/'resource_ledger.json').read_text())
    assert word_cost['single_weight_nodes'] == 2664 and word_cost['word_weight_nodes'] == 10656
    payload = [word_cost['single_nominal_parameter_bytes']]*3+[word_cost['word_nominal_parameter_bytes']]
    assert payload[:3] == [json.loads((FINE/'resource_ledger.json').read_text())['nominal_parameter_payload_bytes']]*3
    csvrows = list(csv.DictReader((OUT/'sampling_curves.csv').open()))
    assert len(d['rows']) == len(csvrows) == 12
    checked = 0
    for method, values, indices in [(0, base['values'], base['bootstrap_seeds']), (1, fine['values'][0], fine['bootstrap_method_seeds'][0]),
                                    (2, fine['values'][1], fine['bootstrap_method_seeds'][1]), (3, word['values'], word['bootstrap_word_seeds'])]:
        for bi, work in enumerate(p['retained_updates_per_weight_bank']):
            point = sum(values[bi, mi, seed] for mi in range(5) for seed in range(10))/50*[100, 1]
            draws = np.zeros((5000, 2))
            for unit in range(5):
                models = base['bootstrap_models'][:, unit]
                samples = indices[:, unit]
                draws += values[bi][models[:, None], samples].sum(1)/50
            draws *= [100, 1]
            intervals = np.quantile(draws, [.025, .975], axis=0).T
            np.testing.assert_allclose(point, a['points'][method, bi], rtol=3e-12, atol=3e-13)
            np.testing.assert_allclose(draws, a['draws'][method, bi], rtol=3e-12, atol=3e-13)
            np.testing.assert_allclose(intervals, a['intervals'][method, bi], rtol=3e-12, atol=3e-13)
            checked += draws.size
            row, exported = d['rows'][method*3+bi], csvrows[method*3+bi]
            k = p['physical_pbits_per_weight'][method]
            assert row['method'] == exported['method'] == LABELS[method]
            expected = {'pbits_per_weight': k, 'samples_per_layer': work//k, 'retained_sample_updates_per_weight_bank': work,
                        'weight_nodes': 2664*k, 'nominal_parameter_payload_bytes': payload[method],
                        'retained_sample_updates_per_full_run': 2664*1250*work, 'warmup_updates_per_full_run': 2664*1250*k,
                        'retained_plus_warmup_updates_per_full_run': 2664*1250*(work+k),
                        'accuracy_percent': point[0], 'mean_CE_nats': point[1],
                        'accuracy_low_percent': intervals[0, 0], 'accuracy_high_percent': intervals[0, 1],
                        'CE_low_nats': intervals[1, 0], 'CE_high_nats': intervals[1, 1]}
            for name, value in expected.items():
                np.testing.assert_allclose([row[name], float(exported[name])], value, rtol=3e-12, atol=3e-13)
    assert checked == 120000
    for name, ci in [('FP32_F64', 0), ('ACT8_INT4_minmax', 4)]:
        reference = sum(base['baseline_metrics'][ci])/5*[100, 1]
        np.testing.assert_allclose(reference, d['reference_mean_metrics'][name], rtol=3e-12, atol=3e-13)
    # Every uncertainty endpoint is visible on the fixed axes.
    assert np.all(a['intervals'][:, :, 0, 0] >= p['accuracy_y_limits_percent'][0])
    assert np.all(a['intervals'][:, :, 0, 1] <= p['accuracy_y_limits_percent'][1])
    assert np.all(a['intervals'][:, :, 1, 0] >= p['CE_y_limits_nats'][0])
    assert np.all(a['intervals'][:, :, 1, 1] <= p['CE_y_limits_nats'][1])
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'points_checked': 12, 'bootstrap_scalar_values_checked': checked,
                                    'all_CSV_fields_intervals_resource_counts_and_references_checked': True,
                                    'scope': 'Separate model-unit accumulation from closed statistic arrays, complete endpoints on axes. New figure only; no new model evaluation, MonteCarlo quality replication or physical measurements.'})
    paths = [OUT/name for name in ['protocol.json', 'data.json', 'figure_arrays.npz', 'sampling_curves.csv',
                                   'native_cnn_finetuning.svg', 'native_cnn_finetuning.png', 'evidence_audit.json']]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all12points_and120000bootstrap_values_checked_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All12figure points/120000bootstrap values andallCSV fields checked', flush=True)


if __name__ == '__main__':
    main()
