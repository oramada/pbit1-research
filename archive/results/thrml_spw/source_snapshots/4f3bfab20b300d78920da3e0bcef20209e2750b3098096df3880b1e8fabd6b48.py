"""Check all figure/CSV units and coverage against the complete BF16 evidence."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw')
OUT = ROOT/'bf16_comparison/figure_evidence'
PARENT = ROOT/'bf16_comparison/evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path
    d = json.loads((OUT/'figure_data.json').read_text())
    assert d['protocol_sha256'] == sha(OUT/'protocol.json')
    s = json.loads((PARENT/'statistics.json').read_text())
    arrays = np.load(PARENT/'bootstrap_arrays.npz')
    bykey = {(r['dataset'], r['method_index'], r['T'], r['control']): r for r in s['contrasts']}
    expected = {(dataset, method, t, control) for dataset in ['fashion', 'cifar'] for method in range(2)
                for t in [64, 256, 1024] for control in ['BF16_weights_F64', 'BF16_Torch_CPU']}
    assert set(bykey) == expected and len(d['rows']) == 24 and len(d['displayed_rows']) == 12
    seen = set()
    for r in d['rows']:
        key = (r['dataset'], r['method_index'], r['samples'], r['BF16_control'])
        assert key not in seen;seen.add(key)
        c = bykey[key]
        assert r['method'] == c['method'] and r['pbits_per_weight'] == 1
        interval = np.quantile(arrays[c['array_key']][:, 0]*100, [.025, .975])
        np.testing.assert_allclose([r['descriptive_95_low_pp'], r['descriptive_95_high_pp']], interval, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose([r['pbit_accuracy_percent'], r['BF16_accuracy_percent'], r['accuracy_difference_pp'], r['CE_difference_nats']],
            [100*c['native_mean'][0], 100*c['BF16_mean'][0], 100*c['difference'][0], c['difference'][1]], rtol=2e-10, atol=2e-12)
    assert seen == expected
    selected = [r for r in d['rows'] if r['BF16_control'] == 'BF16_Torch_CPU']
    assert sorted(d['displayed_rows'], key=lambda r: (r['dataset'], r['method_index'], r['samples'])) == sorted(selected, key=lambda r: (r['dataset'], r['method_index'], r['samples']))
    with (OUT/'BF16_sampling_comparisons.csv').open() as stream:
        csvrows = list(csv.DictReader(stream))
    assert len(csvrows) == 24
    for saved, actual in zip(csvrows, d['rows']):
        assert set(saved) == set(actual)
        for name in actual:
            if isinstance(actual[name], str):
                assert saved[name] == actual[name]
            else:
                assert float(saved[name]) == actual[name]
    assert (OUT/'one_pbit_BF16_sampling.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    svg = (OUT/'one_pbit_BF16_sampling.svg').read_text()
    assert all(label in svg for label in ['Fashion-MNIST', 'Compact RGB CIFAR-10', 'Expected-weight training', 'Sample-aware training'])
    report = ROOT/'SINGLE_PBIT_BF16_SAMPLING_FIGURE.md'
    assert not report.exists()
    report.write_text('''# Sampling required for one-p-bit task quality against BF16

[Vector figure](bf16_comparison/figure_evidence/one_pbit_BF16_sampling.svg) · [PNG](bf16_comparison/figure_evidence/one_pbit_BF16_sampling.png) · [All24comparison CSV](bf16_comparison/figure_evidence/BF16_sampling_comparisons.csv) · [Full BF16 report](SINGLE_PBIT_VS_BF16_REPORT.md)

![Paired task-accuracy differences versus CPU BF16](bf16_comparison/figure_evidence/one_pbit_BF16_sampling.png)

The figure shows all12completed CPU-BF16 contrasts: both training methods, allthree sample budgets, both image tasks. Every BF16-weight-only/F64 comparison is also retained in the24-rowCSV and full report. Error bars are the previously independently verified paired model/within-model MC descriptive95%intervals. Figure andCSV coverage, units and intervals were separately checked; no new stochastic or bootstrap replicas were generated.

Fashion-MNIST's smallT1024observed gap and the larger RGBgap are both visible, alongside all lower-budget losses. Neither a zero-crossing interval nor a small accuracy gap establishes formal BF16 equivalence or per-parameter numerical precision. The extra training epoch budget, fixed consumed test images, deterministic biases/ranges/fieldcommands/ACT8accumulation and physical-cost limits remain. The separateT4096study is still pending and has no points in this figure.

The broader research goal remains active.
''')
    audit = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_sha256': sha(__file__), 'all_CSV_rows_checked': 24, 'displayed_points_checked': 12,
             'accuracy_bootstrap_values_unit_checked': 120000,
             'scope': 'Derived figure/CSV coverage, units and intervals; underlying inference/metric audits retained, no new scientific replicas or equivalence claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    paths = [OUT/n for n in ['protocol.json', 'figure_data.json', 'one_pbit_BF16_sampling.svg', 'one_pbit_BF16_sampling.png', 'BF16_sampling_comparisons.csv', 'evidence_audit.json']]+[report]
    (OUT/'completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {str(path): sha(path) for path in paths}}, indent=2)+'\n')
    print('All24CSVrows/12displayedpoints and120000accuracy bootstrap values unit-checked')


if __name__ == '__main__':
    main()
